"""Hybrid (dense + sparse) retrieval with Pinecone, plus an in-process fallback.

Pinecone hybrid = ONE dotproduct index holding dense + sparse (BM25) vectors per chunk.
Query vectors are scaled:  score = alpha * dense.q + (1 - alpha) * sparse.q   (alpha = HYBRID_ALPHA)
Namespaces:  'general' (public/internal)  and  'restricted' (confidential)  -> physical isolation.
Metadata filters use Pinecone's operator syntax ($eq/$in/$gte/$lte); the local store evaluates the same syntax,
so both backends behave identically.
"""
import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np

from app.auth import Principal
from app.config import settings
from app.logging_setup import log_event
from app.retrieval.corpus import Chunk, tokenize

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*a, **k):
        return lambda f: f

log = logging.getLogger(__name__)


class StoreUnavailable(Exception):
    pass


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense: float | None = None
    sparse: float | None = None
    final: float = 0.0


@dataclass
class SearchResult:
    hits: list[Hit]
    backend: str
    degraded: bool = False
    note: str = ""


# ------------------------------------------------------------------ filters
def build_filter(principal: Principal, department=None, document_type=None, date_from=None, date_to=None) -> dict:
    """The access_level clause comes from the authenticated Principal - the LLM can never widen it."""
    flt: dict = {"access_level": {"$in": sorted(principal.access_levels)}}
    if department:
        flt["department"] = {"$eq": department}
    if document_type:
        flt["document_type"] = {"$eq": document_type}
    rng = {}
    if date_from:
        rng["$gte"] = int(str(date_from).replace("-", ""))
    if date_to:
        rng["$lte"] = int(str(date_to).replace("-", ""))
    if rng:
        flt["created_ts"] = rng
    return flt


def matches(meta: dict, flt: dict) -> bool:
    for field_name, cond in flt.items():
        v = meta.get(field_name)
        for op, arg in (cond if isinstance(cond, dict) else {"$eq": cond}).items():
            if op == "$eq" and v != arg:
                return False
            if op == "$in" and v not in arg:
                return False
            if op == "$gte" and (v is None or v < arg):
                return False
            if op == "$lte" and (v is None or v > arg):
                return False
    return True


# ------------------------------------------------------------------ embeddings
class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        m = self._load()
        fn = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
        return fn()

    async def embed(self, texts: list[str]) -> np.ndarray:
        def _run():
            return self._load().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return await asyncio.to_thread(_run)  # keep the event loop free


def _minmax(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 1e-9 else np.zeros_like(x)


# ------------------------------------------------------------------ local hybrid store
class LocalHybridStore:
    name = "local"

    def __init__(self, embedder: Embedder, alpha: float):
        self.embedder, self.alpha = embedder, alpha
        self.chunks: list[Chunk] = []
        self.dense: np.ndarray | None = None
        self.bm25 = None

    async def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.dense = await self.embedder.embed([c.text for c in chunks])
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    async def search(self, query: str, flt: dict, namespaces: list[str], top_k: int) -> list[Hit]:
        if self.dense is None:
            raise StoreUnavailable("local index not built")
        idx = [i for i, c in enumerate(self.chunks) if c.namespace in namespaces and matches(c.metadata, flt)]
        if not idx:
            return []
        qv = (await self.embedder.embed([query]))[0]
        dense = self.dense[idx] @ qv
        sparse = np.array(self.bm25.get_scores(tokenize(query)))[idx]
        combined = self.alpha * _minmax(dense) + (1 - self.alpha) * _minmax(sparse)
        order = np.argsort(-combined)[:top_k]
        return [Hit(self.chunks[idx[i]], float(combined[i]), float(dense[i]), float(sparse[i])) for i in order]


# ------------------------------------------------------------------ pinecone hybrid store
class PineconeHybridStore:
    name = "pinecone"

    def __init__(self, embedder: Embedder, alpha: float):
        import time
        from pinecone import Pinecone, ServerlessSpec
        self.embedder, self.alpha = embedder, alpha
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        names = [i.name for i in self.pc.list_indexes()]
        if settings.pinecone_index not in names:
            self.pc.create_index(name=settings.pinecone_index, dimension=embedder.dim, metric="dotproduct",
                                 spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region))
            for _ in range(30):  # wait until the new index is ready
                if self.pc.describe_index(settings.pinecone_index).status["ready"]:
                    break
                time.sleep(2)
        self.index = self.pc.Index(settings.pinecone_index)
        self.bm25 = None

    def _encoder(self):
        if self.bm25 is None:
            from pinecone_text.sparse import BM25Encoder
            try:
                self.bm25 = BM25Encoder().load(settings.bm25_path)
            except Exception as exc:
                raise StoreUnavailable(f"BM25 params missing - run `python -m scripts.ingest` ({exc})")
        return self.bm25

    async def upsert(self, chunks: list[Chunk]) -> None:
        from pinecone_text.sparse import BM25Encoder
        enc = BM25Encoder()
        await asyncio.to_thread(enc.fit, [c.text for c in chunks])
        enc.dump(settings.bm25_path)
        self.bm25 = enc
        dense = await self.embedder.embed([c.text for c in chunks])
        by_ns: dict[str, list] = {}
        for c, d in zip(chunks, dense):
            sp = enc.encode_documents(c.text)
            meta = {**c.metadata, "doc_id": c.doc_id, "title": c.title, "text": c.text}
            by_ns.setdefault(c.namespace, []).append({
                "id": c.id, "values": (d * self.alpha).tolist(),
                "sparse_values": {"indices": sp["indices"], "values": [v * (1 - self.alpha) for v in sp["values"]]},
                "metadata": meta})
        for ns, vecs in by_ns.items():
            for i in range(0, len(vecs), 100):
                await asyncio.to_thread(self.index.upsert, vectors=vecs[i:i + 100], namespace=ns)

    async def search(self, query: str, flt: dict, namespaces: list[str], top_k: int) -> list[Hit]:
        enc = self._encoder()
        qd = (await self.embedder.embed([query]))[0]
        qs = enc.encode_queries(query)
        vec = (qd * self.alpha).tolist()
        sparse = {"indices": qs["indices"], "values": [v * (1 - self.alpha) for v in qs["values"]]}
        hits: list[Hit] = []
        for ns in namespaces:
            res = await asyncio.to_thread(self.index.query, vector=vec, sparse_vector=sparse, top_k=top_k,
                                          filter=flt, namespace=ns, include_metadata=True)
            for m in res["matches"]:
                md = dict(m["metadata"])
                text, title, doc_id = md.pop("text"), md.pop("title"), md.pop("doc_id")
                hits.append(Hit(Chunk(m["id"], doc_id, title, text, md), float(m["score"])))
        return sorted(hits, key=lambda h: -h.score)[:top_k]


# ------------------------------------------------------------------ facade with graceful degradation
class ResilientStore:
    """Tries Pinecone; on ANY failure falls back to the local hybrid index and flags the result as degraded."""

    def __init__(self, local: LocalHybridStore, primary: PineconeHybridStore | None):
        self.local, self.primary = local, primary

    @traceable(run_type="retriever", name="hybrid_search")
    async def search(self, principal: Principal, query: str, top_k: int = 8, department=None,
                     document_type=None, date_from=None, date_to=None) -> SearchResult:
        flt = build_filter(principal, department, document_type, date_from, date_to)
        if self.primary:
            try:
                hits = await asyncio.wait_for(self.primary.search(query, flt, principal.namespaces, top_k), 15)
                return SearchResult(hits, "pinecone")
            except Exception as exc:
                log_event(log, "vector_db_failure", level=logging.WARNING, error=str(exc)[:200])
                hits = await self.local.search(query, flt, principal.namespaces, top_k)
                return SearchResult(hits, "local-fallback", True, f"Pinecone unavailable: {str(exc)[:80]}")
        return SearchResult(await self.local.search(query, flt, principal.namespaces, top_k), "local")

    async def rebuild(self, chunks: list[Chunk]) -> None:
        await self.local.build(chunks)
        if self.primary:
            await self.primary.upsert(chunks)


def rerank(query: str, hits: list[Hit], top_k: int, max_per_doc: int = 2) -> list[Hit]:
    """Lightweight reranker: normalised hybrid score + title-term boost, with per-document diversity cap.
    (A cross-encoder such as bge-reranker is the production upgrade; the interface stays the same.)"""
    if not hits:
        return []
    q = set(tokenize(query))
    top = max(h.score for h in hits) or 1.0
    for h in hits:
        title_overlap = len(q & set(tokenize(h.chunk.title))) / (len(q) or 1)
        h.final = h.score / top + 0.25 * title_overlap
    out, per_doc = [], {}
    for h in sorted(hits, key=lambda h: -h.final):
        if per_doc.get(h.chunk.doc_id, 0) < max_per_doc:
            per_doc[h.chunk.doc_id] = per_doc.get(h.chunk.doc_id, 0) + 1
            out.append(h)
        if len(out) == top_k:
            break
    return out
