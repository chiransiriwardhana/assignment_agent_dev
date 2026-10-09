"""Index documents into Pinecone (namespaces general / restricted) and fit the BM25 encoder.
Run: python -m scripts.ingest
"""
import asyncio

from app.config import settings
from app.logging_setup import setup_logging
from app.retrieval.corpus import Corpus
from app.retrieval.store import Embedder, PineconeHybridStore


async def main():
    setup_logging()
    corpus = Corpus(settings.docs_dir)
    if not settings.pinecone_api_key:
        print("PINECONE_API_KEY not set - the API will use the local hybrid index. Nothing to ingest.")
        return
    store = PineconeHybridStore(Embedder(settings.embedding_model), settings.hybrid_alpha)
    await store.upsert(corpus.chunks)
    print(f"Indexed {len(corpus.docs)} documents / {len(corpus.chunks)} chunks into '{settings.pinecone_index}'.")


if __name__ == "__main__":
    asyncio.run(main())
