"""Ingest RevenueCat documentation into the knowledge base.

Uses Firecrawl to crawl the official docs, chunks pages by token count,
generates embeddings via Voyage-3 (with fallback), and stores everything
in PostgreSQL with pgvector.

CLI usage:
    python scripts/ingest_docs.py --full     # Full crawl of all doc URLs
    python scripts/ingest_docs.py --check    # Show current DB stats
    python scripts/ingest_docs.py --update   # Re-ingest only changed pages
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from firecrawl import FirecrawlApp

from config.settings import settings
from database.db import get_supabase, insert_knowledge, _get_engine
from tools.search_docs import embed, embed_batch

# ---------------------------------------------------------------------------
# Token-aware chunking
# ---------------------------------------------------------------------------

# Rough approximation: 1 token ~ 4 chars for English text.
# This keeps us safely under the configured token limits.
CHARS_PER_TOKEN = 4
MAX_CHUNK_CHARS = settings.chunk_max_tokens * CHARS_PER_TOKEN  # 3200
OVERLAP_CHARS = settings.chunk_overlap_tokens * CHARS_PER_TOKEN  # 400


def _chunk_text(text: str) -> list[str]:
    """Split *text* into overlapping chunks of ~800 tokens each."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            # Look for a paragraph break within the last 20% of the chunk
            search_start = end - MAX_CHUNK_CHARS // 5
            para_break = text.rfind("\n\n", search_start, end)
            if para_break > start:
                end = para_break + 2  # include the double newline
            else:
                # Fall back to sentence boundary
                sentence_break = text.rfind(". ", search_start, end)
                if sentence_break > start:
                    end = sentence_break + 2
        chunks.append(text[start:end].strip())
        start = end - OVERLAP_CHARS
    return [c for c in chunks if c]


def _content_hash(text: str) -> str:
    """SHA-256 hex digest of the text, used for change detection."""
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Firecrawl helpers
# ---------------------------------------------------------------------------


def _crawl_url(app: FirecrawlApp, url: str) -> list[dict]:
    """Crawl a single URL and return page data dicts."""
    print(f"  Crawling {url} ...")
    result = app.crawl_url(
        url,
        params={
            "limit": 50,
            "scrapeOptions": {"formats": ["markdown"]},
        },
    )
    if isinstance(result, dict):
        return result.get("data", [])
    return list(result) if result else []


def _scrape_url(app: FirecrawlApp, url: str) -> dict | None:
    """Scrape a single page (no recursive crawl)."""
    print(f"  Scraping {url} ...")
    try:
        result = app.scrape_url(url, params={"formats": ["markdown"]})
        return result if result else None
    except Exception as exc:
        print(f"    Warning: failed to scrape {url}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Ingestion modes
# ---------------------------------------------------------------------------

BATCH_SIZE = 16  # embedding batch size


async def _ingest_pages(pages: list[dict]) -> int:
    """Chunk, embed, and store a list of page dicts. Returns chunk count."""
    total = 0
    for page in pages:
        content = page.get("markdown", "") or page.get("content", "")
        metadata = page.get("metadata", {})
        url = metadata.get("sourceURL", metadata.get("url", ""))
        section = metadata.get("title", "")

        if not content.strip():
            continue

        chunks = _chunk_text(content)
        # Embed in batches
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start : batch_start + BATCH_SIZE]
            embeddings = embed_batch(batch)
            for chunk_text, emb in zip(batch, embeddings):
                await insert_knowledge(
                    content=chunk_text,
                    embedding=emb,
                    source_url=url,
                    section=section,
                )
                total += 1
        print(f"    Ingested {len(chunks)} chunks from {url or '(unknown)'}")
    return total


async def full_ingest() -> int:
    """Full crawl of all configured doc URLs."""
    app = FirecrawlApp(api_key=settings.firecrawl_api_key)
    all_pages: list[dict] = []

    for url in settings.docs_urls:
        pages = _crawl_url(app, url)
        all_pages.extend(pages)

    print(f"\nCrawled {len(all_pages)} pages total. Ingesting ...")
    total = await _ingest_pages(all_pages)
    print(f"\nDone. {total} chunks ingested into knowledge_base.")
    return total


async def update_ingest() -> int:
    """Re-ingest only pages whose content has changed.

    Compares a SHA-256 hash of each page's markdown against what is already
    stored (by source_url). New or changed pages are re-ingested; unchanged
    pages are skipped.
    """
    app = FirecrawlApp(api_key=settings.firecrawl_api_key)
    sb = get_supabase()

    # Build a set of existing (url -> hash) from the DB.
    # We store the hash of the first chunk's content per URL as a proxy.
    existing_resp = (
        sb.table("knowledge_base")
        .select("source_url, content")
        .execute()
    )
    existing_hashes: dict[str, set[str]] = {}
    for row in existing_resp.data:
        existing_hashes.setdefault(row["source_url"], set()).add(
            _content_hash(row["content"])
        )

    changed_pages: list[dict] = []
    for url in settings.docs_urls:
        page = _scrape_url(app, url)
        if not page:
            continue
        content = page.get("markdown", "") or page.get("content", "")
        if not content.strip():
            continue
        page_url = page.get("metadata", {}).get("sourceURL", url)
        first_chunk = _chunk_text(content)[0] if content else ""
        h = _content_hash(first_chunk)
        if page_url in existing_hashes and h in existing_hashes[page_url]:
            print(f"    Unchanged: {page_url}")
            continue
        changed_pages.append(page)

    if not changed_pages:
        print("All pages are up-to-date. Nothing to ingest.")
        return 0

    print(f"\n{len(changed_pages)} changed page(s) detected. Ingesting ...")
    total = await _ingest_pages(changed_pages)
    print(f"\nDone. {total} chunks ingested (update mode).")
    return total


async def check_status() -> None:
    """Print current knowledge base statistics."""
    sb = get_supabase()

    kb = sb.table("knowledge_base").select("id", count="exact").execute()
    mem = sb.table("memory_nuggets").select("id", count="exact").execute()

    # Distinct source URLs
    urls_resp = sb.table("knowledge_base").select("source_url").execute()
    unique_urls = {r["source_url"] for r in urls_resp.data} if urls_resp.data else set()

    print("Knowledge Base Status")
    print("---------------------")
    print(f"  Total chunks:         {kb.count or 0}")
    print(f"  Unique source URLs:   {len(unique_urls)}")
    print(f"  Memory nuggets:       {mem.count or 0}")
    if unique_urls:
        print("\n  Indexed URLs:")
        for u in sorted(unique_urls):
            print(f"    - {u}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RevenueCat documentation ingestion pipeline"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--full", action="store_true", help="Full crawl and ingest all docs"
    )
    group.add_argument(
        "--check", action="store_true", help="Check current DB status"
    )
    group.add_argument(
        "--update",
        action="store_true",
        help="Re-ingest only changed pages",
    )

    args = parser.parse_args()

    if args.full:
        asyncio.run(full_ingest())
    elif args.update:
        asyncio.run(update_ingest())
    elif args.check:
        asyncio.run(check_status())


if __name__ == "__main__":
    main()
