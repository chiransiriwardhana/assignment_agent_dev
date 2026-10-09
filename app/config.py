"""Central configuration. Everything is overridable via environment variables / .env."""
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional at import time (e.g. in unit tests)
    pass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # LLM: open-source Llama models served by Groq
    groq_api_key: str = _env("GROQ_API_KEY", "")
    primary_model: str = _env("GROQ_PRIMARY_MODEL", "openai/gpt-oss-120b")
    fast_model: str = _env("GROQ_FAST_MODEL", "openai/gpt-oss-120b")

    # Retrieval
    pinecone_api_key: str = _env("PINECONE_API_KEY", "")
    pinecone_index: str = _env("PINECONE_INDEX", "crest-kb")
    pinecone_cloud: str = _env("PINECONE_CLOUD", "aws")
    pinecone_region: str = _env("PINECONE_REGION", "us-east-1")
    embedding_model: str = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    hybrid_alpha: float = float(_env("HYBRID_ALPHA", "0.6"))  # 1.0 = dense only, 0.0 = sparse only
    docs_dir: str = _env("DOCS_DIR", "data/docs")
    bm25_path: str = _env("BM25_PATH", "data/bm25.json")

    # Rate limiting (token bucket, per user, per role)
    rate_capacity: dict = field(default_factory=lambda: {
        "viewer": int(_env("RATE_CAPACITY_VIEWER", "5")),
        "analyst": int(_env("RATE_CAPACITY_ANALYST", "10")),
        "admin": int(_env("RATE_CAPACITY_ADMIN", "20")),
    })
    rate_refill_per_sec: float = float(_env("RATE_REFILL_PER_SEC", "0.2"))

    # Runtime limits
    tool_timeout_sec: float = float(_env("TOOL_TIMEOUT_SEC", "25"))
    turn_timeout_sec: float = float(_env("TURN_TIMEOUT_SEC", "120"))
    demo_password: str = _env("DEMO_PASSWORD", "demo123")
    memory_dir: str = _env("MEMORY_DIR", "data/memory")
    feedback_path: str = _env("FEEDBACK_PATH", "data/feedback.jsonl")


settings = Settings()
