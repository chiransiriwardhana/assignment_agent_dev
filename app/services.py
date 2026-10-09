"""Service container built once at startup (dependency wiring in one place)."""
import logging
from dataclasses import dataclass, field

from app.config import settings
from app.logging_setup import log_event
from app.memory import MemoryManager
from app.rate_limit import RateLimiter
from app.retrieval.corpus import Corpus
from app.retrieval.store import Embedder, LocalHybridStore, PineconeHybridStore, ResilientStore
from app.tools import ToolExecutor, build_executor

log = logging.getLogger(__name__)


@dataclass
class Services:
    corpus: Corpus
    store: ResilientStore
    executor: ToolExecutor
    memory: MemoryManager
    limiter: RateLimiter
    graph: object = None
    pending: dict = field(default_factory=dict)  # "<user>:<session>" -> thread_id awaiting human approval


_services: Services | None = None


def get_services() -> Services:
    if _services is None:
        raise RuntimeError("services not initialised")
    return _services


async def build_services() -> Services:
    global _services
    from app.agents.graph import build_graph  # local import avoids a cycle (graph -> get_services)

    corpus = Corpus(settings.docs_dir)
    embedder = Embedder(settings.embedding_model)
    local = LocalHybridStore(embedder, settings.hybrid_alpha)
    await local.build(corpus.chunks)

    primary = None
    if settings.pinecone_api_key:
        try:
            primary = PineconeHybridStore(embedder, settings.hybrid_alpha)
        except Exception as exc:  # run in degraded mode instead of refusing to start
            log_event(log, "pinecone_init_failed", level=logging.ERROR, error=str(exc)[:200])

    _services = Services(
        corpus=corpus, store=ResilientStore(local, primary), executor=build_executor(),
        memory=MemoryManager(), limiter=RateLimiter(settings.rate_capacity, settings.rate_refill_per_sec),
    )
    _services.graph = build_graph()
    log_event(log, "services_ready", docs=len(corpus.docs), chunks=len(corpus.chunks),
              backend="pinecone" if primary else "local")
    return _services
