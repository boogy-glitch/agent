"""Ingest RevenueCat documentation into the knowledge base.

Uses Firecrawl to crawl docs.revenuecat.com, chunks the pages,
generates embeddings, and stores them in PostgreSQL via pgvector.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from firecrawl import FirecrawlApp

from config.settings import settings
from database.db import insert_knowledge
from tools.search_docs import _embed

DOCS_URL = "https://docs.revenuecat.com"
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


async def ingest() -> int:
    """Crawl RevenueCat docs and ingest into the knowledge base.

    Returns the total number of chunks inserted.
    """
    app = FirecrawlApp(api_key=settings.firecrawl_api_key)

    print(f"Crawling {DOCS_URL} ...")
    result = app.crawl_url(
        DOCS_URL,
        params={
            "limit": 200,
            "scrapeOptions": {"formats": ["markdown"]},
        },
    )

    total = 0
    pages = result.get("data", []) if isinstance(result, dict) else result
    for page in pages:
        content = page.get("markdown", "") or page.get("content", "")
        url = page.get("metadata", {}).get("sourceURL", DOCS_URL)
        section = page.get("metadata", {}).get("title", "")

        if not content.strip():
            continue

        chunks = _chunk_text(content)
        for chunk in chunks:
            embedding = _embed(chunk)
            await insert_knowledge(
                content=chunk,
                embedding=embedding,
                source_url=url,
                section=section,
            )
            total += 1
        print(f"  Ingested {len(chunks)} chunks from {url}")

    print(f"Done. {total} total chunks ingested.")
    return total


if __name__ == "__main__":
    asyncio.run(ingest())
