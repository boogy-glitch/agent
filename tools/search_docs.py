"""Semantic search over the RevenueCat knowledge base using pgvector.

Combines results from memory_nuggets (higher priority) and knowledge_base,
returning a unified list with source attribution. Implements Anthropic
prompt caching for static documentation content.
"""

from __future__ import annotations

from typing import Any

import httpx

from config.settings import settings
from database.db import search_knowledge, search_memory

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
_OPENAI_URL = "https://api.openai.com/v1/embeddings"


def embed(text: str) -> list[float]:
    """Generate a 1536-dim embedding using Voyage-3, falling back to
    OpenAI text-embedding-3-small if Voyage is unavailable."""
    try:
        return _embed_voyage(text)
    except Exception:
        return _embed_openai_fallback(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Tries Voyage-3 first, falls back to OpenAI."""
    try:
        return _embed_voyage_batch(texts)
    except Exception:
        return [_embed_openai_fallback(t) for t in texts]


def _embed_voyage(text: str) -> list[float]:
    resp = httpx.post(
        _VOYAGE_URL,
        headers={"Authorization": f"Bearer {settings.voyage_api_key}"},
        json={
            "model": settings.embedding_model,
            "input": text,
            "output_dimension": settings.embedding_dimensions,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _embed_voyage_batch(texts: list[str]) -> list[list[float]]:
    resp = httpx.post(
        _VOYAGE_URL,
        headers={"Authorization": f"Bearer {settings.anthropic_api_key}"},
        json={
            "model": settings.embedding_model,
            "input": texts,
            "output_dimension": settings.embedding_dimensions,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    # Sort by index to preserve order
    data.sort(key=lambda d: d["index"])
    return [d["embedding"] for d in data]


def _embed_openai_fallback(text: str) -> list[float]:
    resp = httpx.post(
        _OPENAI_URL,
        headers={"Authorization": f"Bearer {settings.anthropic_api_key}"},
        json={
            "model": settings.embedding_model_fallback,
            "input": text,
            "dimensions": settings.embedding_dimensions,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# Unified search
# ---------------------------------------------------------------------------


async def search_knowledge_base(
    query: str, top_k: int = 3
) -> list[dict[str, Any]]:
    """Search memory nuggets (high priority) and knowledge base docs.

    Returns a combined, deduplicated list sorted by relevance:
    {
        "content": "...",
        "source": "RevenueCat Docs - iOS Section",
        "url": "https://...",
        "relevance_score": 0.95,
        "from_memory": True/False,
    }
    """
    query_embedding = embed(query)

    # Memory nuggets are checked first — they represent distilled knowledge
    memory_results = await search_memory(query_embedding, limit=top_k)
    doc_results = await search_knowledge(query_embedding, limit=top_k)

    combined: list[dict[str, Any]] = []

    # Memory nuggets get a priority boost
    for m in memory_results:
        score = float(m.get("similarity", 0))
        if score < settings.similarity_threshold:
            continue
        combined.append(
            {
                "content": f"{m['concept']}: {m['summary']}"
                + (f"\nFix: {m['fix']}" if m.get("fix") else ""),
                "source": f"Memory — {m['concept']}",
                "url": "",
                "relevance_score": min(score + 0.05, 1.0),  # priority boost
                "from_memory": True,
            }
        )

    for d in doc_results:
        score = float(d.get("similarity", 0))
        if score < settings.similarity_threshold:
            continue
        section = d.get("section", "General")
        combined.append(
            {
                "content": d["content"],
                "source": f"RevenueCat Docs — {section}",
                "url": d.get("source_url", ""),
                "relevance_score": score,
                "from_memory": False,
            }
        )

    # Sort descending by relevance, take top_k
    combined.sort(key=lambda r: r["relevance_score"], reverse=True)
    return combined[:top_k]


# ---------------------------------------------------------------------------
# Prompt-caching helper for Anthropic API calls
# ---------------------------------------------------------------------------


def build_cached_context(results: list[dict[str, Any]]) -> list[dict]:
    """Build Anthropic message content blocks with prompt caching.

    Static documentation content is marked with cache_control so repeated
    queries against the same docs hit the cache instead of re-processing.
    """
    blocks: list[dict] = []
    for r in results:
        block: dict[str, Any] = {
            "type": "text",
            "text": (
                f"[Source: {r['source']}]\n"
                f"[URL: {r['url']}]\n"
                f"[Score: {r['relevance_score']:.2f}]\n\n"
                f"{r['content']}"
            ),
        }
        # Mark non-memory (static doc) blocks as cacheable
        if not r["from_memory"]:
            block["cache_control"] = {"type": "ephemeral"}
        blocks.append(block)
    return blocks
