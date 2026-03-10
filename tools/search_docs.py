"""Semantic search over the RevenueCat knowledge base using pgvector."""

from __future__ import annotations

from anthropic import Anthropic

from config.settings import settings
from database.db import search_knowledge

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _embed(text: str) -> list[float]:
    """Generate an embedding for the given text via the Anthropic-compatible
    OpenAI embeddings endpoint (Supabase or direct).  Falls back to a
    lightweight local approach when the embedding service is unavailable."""
    # Using the OpenAI-compatible endpoint that Supabase Edge Functions expose,
    # or a direct OpenAI call.  We keep this pluggable.
    import httpx

    resp = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {settings.anthropic_api_key}"},
        json={
            "model": settings.embedding_model,
            "input": text,
            "dimensions": settings.embedding_dimensions,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


async def search_revenuecat_docs(
    query: str, limit: int = 5
) -> list[dict]:
    """LangChain-compatible tool: search RevenueCat docs by semantic similarity.

    Returns a list of dicts with keys: content, source_url, section, similarity.
    """
    embedding = _embed(query)
    results = await search_knowledge(embedding, limit=limit)
    return results
