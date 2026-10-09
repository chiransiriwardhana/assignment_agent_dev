"""Document corpus: the source of truth for documents, chunking and 'RLM-style' exploration.

The vector DB is an index over this corpus. Keeping the corpus separately lets the Research agent
explore (list / grep / select sections) without loading whole documents into an LLM context, and gives
us a degraded-mode data source if the vector DB is down.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from app.auth import Principal


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def namespace_for(access_level: str) -> str:
    return "restricted" if access_level == "confidential" else "general"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    body: str
    metadata: dict


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    metadata: dict

    @property
    def namespace(self) -> str:
        return namespace_for(self.metadata["access_level"])


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    _, header, body = raw.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    doc_id, title = meta.pop("id"), meta.pop("title")
    meta["created_ts"] = int(meta["created_date"].replace("-", ""))  # numeric => range filters in Pinecone
    return Document(doc_id, title, body.strip(), meta)


def chunk_document(doc: Document, max_chars: int = 900) -> list[Chunk]:
    """Split on '## ' sections, then pack paragraphs up to max_chars."""
    chunks: list[Chunk] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            n = len(chunks)
            chunks.append(Chunk(f"{doc.id}#{n}", doc.id, doc.title, f"{doc.title}\n{buf.strip()}", doc.metadata))
        buf = ""

    for section in re.split(r"\n(?=## )", doc.body):
        for para in section.split("\n\n"):
            if buf and len(buf) + len(para) > max_chars:
                flush()
            buf += ("\n\n" if buf else "") + para
        flush()
    return chunks


class Corpus:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.docs: dict[str, Document] = {}
        self.chunks: list[Chunk] = []
        self.reload()

    def reload(self) -> None:
        self.docs = {}
        for p in sorted(Path(self.docs_dir).glob("*.md")):
            d = parse_document(p)
            self.docs[d.id] = d
        self.chunks = [c for d in self.docs.values() for c in chunk_document(d)]

    # ---- access-controlled exploration primitives (used by the RLM sandbox and the analysis tool)
    def list_docs(self, principal: Principal, department=None, document_type=None,
                  date_from=None, date_to=None, keyword=None) -> list[dict]:
        out = []
        for d in self.docs.values():
            m = d.metadata
            if not principal.can_see(m):
                continue
            if department and m["department"] != department:
                continue
            if document_type and m["document_type"] != document_type:
                continue
            if date_from and m["created_date"] < str(date_from):
                continue
            if date_to and m["created_date"] > str(date_to):
                continue
            if keyword and keyword.lower() not in (d.title + " " + d.body).lower():
                continue
            out.append({"id": d.id, "title": d.title, "department": m["department"],
                        "document_type": m["document_type"], "access_level": m["access_level"],
                        "date": m["created_date"]})
        return out

    def contains(self, principal: Principal, term: str, doc_ids: list[str] | None = None) -> list[str]:
        visible = {d["id"] for d in self.list_docs(principal)}
        pool = visible if doc_ids is None else visible & set(doc_ids)
        t = term.lower()
        return [i for i in pool if t in (self.docs[i].title + " " + self.docs[i].body).lower()]

    def select_sections(self, doc_id: str, question: str, budget: int = 1800) -> str:
        """Targeted retrieval inside one document: keep the chunks most related to the question."""
        q = set(tokenize(question)) - {"the", "a", "of", "and", "to", "in", "for", "all", "is"}
        mine = [c for c in self.chunks if c.doc_id == doc_id]
        ranked = sorted(mine, key=lambda c: -len(q & set(tokenize(c.text))))
        picked, used = [], 0
        for c in ranked:
            if used + len(c.text) > budget and picked:
                break
            picked.append(c)
            used += len(c.text)
        picked.sort(key=lambda c: int(c.id.split("#")[1]))
        return "\n".join(c.text for c in picked)
