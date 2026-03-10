"""Database connection and query helpers using Supabase + SQLAlchemy."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from supabase import Client, create_client

from config.settings import settings

# ---------------------------------------------------------------------------
# Supabase client (used for auth, storage, and simple CRUD)
# ---------------------------------------------------------------------------

_supabase: Client | None = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(settings.supabase_url, settings.supabase_key)
    return _supabase


# ---------------------------------------------------------------------------
# Async SQLAlchemy engine (used for pgvector queries)
# ---------------------------------------------------------------------------

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        # Build a Postgres connection string from the Supabase URL.
        # Supabase exposes a direct Postgres connection on port 5432.
        db_url = (
            settings.supabase_url
            .replace("https://", "postgresql+asyncpg://postgres:postgres@")
            + ":5432/postgres"
        )
        _engine = create_async_engine(db_url, pool_size=5)
    return _engine


# ---------------------------------------------------------------------------
# Knowledge-base helpers
# ---------------------------------------------------------------------------


async def search_knowledge(
    embedding: list[float], limit: int = 5, threshold: float | None = None
) -> list[dict[str, Any]]:
    """Return the closest documentation chunks by cosine similarity."""
    threshold = threshold or settings.similarity_threshold
    engine = _get_engine()
    async with AsyncSession(engine) as session:
        result = await session.execute(
            text(
                """
                SELECT id, content, source_url, section,
                       1 - (embedding <=> :emb::vector) AS similarity
                FROM knowledge_base
                WHERE 1 - (embedding <=> :emb::vector) >= :threshold
                ORDER BY similarity DESC
                LIMIT :lim
                """
            ),
            {"emb": str(embedding), "threshold": threshold, "lim": limit},
        )
        return [dict(row._mapping) for row in result]


async def insert_knowledge(
    content: str,
    embedding: list[float],
    source_url: str,
    section: str,
) -> str:
    """Insert a documentation chunk and return its id."""
    row_id = str(uuid.uuid4())
    engine = _get_engine()
    async with AsyncSession(engine) as session:
        await session.execute(
            text(
                """
                INSERT INTO knowledge_base (id, content, embedding, source_url, section)
                VALUES (:id, :content, :emb::vector, :url, :section)
                """
            ),
            {
                "id": row_id,
                "content": content,
                "emb": str(embedding),
                "url": source_url,
                "section": section,
            },
        )
        await session.commit()
    return row_id


# ---------------------------------------------------------------------------
# Interaction helpers
# ---------------------------------------------------------------------------


async def upsert_interaction(data: dict[str, Any]) -> str:
    """Insert or update a tweet interaction. Returns the row id."""
    sb = get_supabase()
    data.setdefault("id", str(uuid.uuid4()))
    sb.table("interactions").upsert(data).execute()
    return data["id"]


async def get_pending_interactions() -> list[dict[str, Any]]:
    sb = get_supabase()
    resp = (
        sb.table("interactions")
        .select("*")
        .eq("status", "PENDING_APPROVAL")
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data


async def update_interaction_status(interaction_id: str, status: str) -> None:
    sb = get_supabase()
    sb.table("interactions").update({"status": status}).eq(
        "id", interaction_id
    ).execute()


async def get_uncompacted_interactions(limit: int = 10) -> list[dict[str, Any]]:
    """Return successful interactions that haven't been compacted yet."""
    sb = get_supabase()
    resp = (
        sb.table("interactions")
        .select("*")
        .eq("status", "APPROVED")
        .eq("compacted", False)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return resp.data


async def mark_interactions_compacted(interaction_ids: list[str]) -> None:
    """Mark a batch of interactions as compacted."""
    sb = get_supabase()
    for iid in interaction_ids:
        sb.table("interactions").update({"compacted": True}).eq(
            "id", iid
        ).execute()


# ---------------------------------------------------------------------------
# Memory-nugget helpers
# ---------------------------------------------------------------------------


async def search_memory(
    embedding: list[float], limit: int = 5
) -> list[dict[str, Any]]:
    engine = _get_engine()
    async with AsyncSession(engine) as session:
        result = await session.execute(
            text(
                """
                SELECT id, concept, summary, fix, importance, usage_count,
                       1 - (embedding <=> :emb::vector) AS similarity
                FROM memory_nuggets
                ORDER BY (embedding <=> :emb::vector) ASC
                LIMIT :lim
                """
            ),
            {"emb": str(embedding), "lim": limit},
        )
        return [dict(row._mapping) for row in result]


async def upsert_memory_nugget(data: dict[str, Any]) -> str:
    sb = get_supabase()
    data.setdefault("id", str(uuid.uuid4()))
    sb.table("memory_nuggets").upsert(data).execute()
    return data["id"]


async def increment_memory_usage(nugget_id: str) -> None:
    sb = get_supabase()
    sb.rpc("increment_usage_count", {"row_id": nugget_id}).execute()


# ---------------------------------------------------------------------------
# Insight-report helpers
# ---------------------------------------------------------------------------


async def insert_insight_report(
    week_start: date, content: str, pain_points: list[dict]
) -> str:
    row_id = str(uuid.uuid4())
    sb = get_supabase()
    sb.table("insight_reports").insert(
        {
            "id": row_id,
            "week_start": week_start.isoformat(),
            "content": content,
            "pain_points": pain_points,
        }
    ).execute()
    return row_id


async def get_latest_report() -> dict[str, Any] | None:
    sb = get_supabase()
    resp = (
        sb.table("insight_reports")
        .select("*")
        .order("week_start", desc=True)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None
