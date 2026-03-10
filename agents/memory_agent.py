"""Always-On Memory compaction agent.

Periodically scans successful interactions, extracts reusable knowledge
into compact *memory nuggets* with embeddings, and serves as a Token
Efficiency Router that decides whether to use cached memory, partial
RAG, or full RAG based on similarity scores.

~200 tokens per compaction vs ~2000 for raw lookup = 90% savings.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import schedule

from anthropic import Anthropic

from config.settings import settings
from database.db import (
    get_uncompacted_interactions,
    increment_memory_usage,
    mark_interactions_compacted,
    search_memory,
    upsert_memory_nugget,
)
from tools.search_docs import embed, search_knowledge_base

logger = logging.getLogger("memory_agent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)

# System prompt for the extraction call — kept short for Haiku efficiency
_EXTRACTION_SYSTEM = (
    "You are a knowledge extractor. Extract the core technical insight "
    "from these developer interactions into a 'Memory Nugget'. "
    "Return JSON only: {\"concept\": \"...\", \"summary\": \"...\", "
    "\"fix\": \"...\", \"importance\": 0.0-1.0}"
)


class MemoryAgent:
    """Always-On Memory system with compaction and token-efficient routing."""

    def __init__(
        self,
        anthropic_client: Anthropic | None = None,
    ):
        self._client = anthropic_client or Anthropic(
            api_key=settings.anthropic_api_key,
        )
        self._model = settings.claude_model_haiku

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def compact_interactions(self) -> int:
        """Compact unprocessed successful interactions into memory nuggets.

        Triggered when >= 5 uncompacted APPROVED interactions exist.
        Returns the number of new nuggets created.
        """
        interactions = await get_uncompacted_interactions(limit=10)

        if len(interactions) < settings.memory_compaction_min_interactions:
            logger.info(
                "Only %d uncompacted interactions (need %d). Skipping.",
                len(interactions),
                settings.memory_compaction_min_interactions,
            )
            return 0

        # Take the first 5 for this compaction cycle
        batch = interactions[: settings.memory_compaction_min_interactions]

        block = "\n---\n".join(
            f"Q: {ix['tweet_text']}\nA: {ix['draft_reply']}"
            + (f"\nCode: {ix['code_snippet']}" if ix.get("code_snippet") else "")
            for ix in batch
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=_EXTRACTION_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract memory nuggets from these interactions. "
                        "Return a JSON array of nugget objects.\n\n"
                        f"{block}"
                    ),
                }
            ],
        )

        raw = response.content[0].text.strip()
        # Handle both single object and array responses
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown fences
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse LLM response as JSON")
                    return 0
            else:
                logger.warning("Failed to parse LLM response as JSON")
                return 0

        nuggets = parsed if isinstance(parsed, list) else [parsed]

        created = 0
        for n in nuggets:
            concept = n.get("concept", "")
            summary = n.get("summary", "")
            if not concept or not summary:
                continue
            emb = embed(f"{concept} {summary}")
            await upsert_memory_nugget(
                {
                    "concept": concept,
                    "summary": summary,
                    "fix": n.get("fix", ""),
                    "embedding": emb,
                    "importance": float(n.get("importance", 0.5)),
                }
            )
            created += 1

        # Mark interactions as compacted
        ids = [ix["id"] for ix in batch]
        await mark_interactions_compacted(ids)

        logger.info(
            "Compacted %d interactions -> %d nuggets at %s",
            len(batch),
            created,
            datetime.now(timezone.utc).isoformat(),
        )
        return created

    # ------------------------------------------------------------------
    # Memory search
    # ------------------------------------------------------------------

    async def search_memories(self, query: str) -> list[dict]:
        """Vector similarity search in memory_nuggets.

        Returns top 2 most relevant memories and increments their usage_count.
        """
        query_embedding = embed(query)
        results = await search_memory(query_embedding, limit=2)

        # Increment usage counters for retrieved memories
        for r in results:
            try:
                await increment_memory_usage(r["id"])
                r["usage_count"] = r.get("usage_count", 0) + 1
            except Exception:
                pass  # non-critical

        return results

    # ------------------------------------------------------------------
    # Token Efficiency Router
    # ------------------------------------------------------------------

    async def get_context_for_query(self, query: str) -> str:
        """Decide how much context to retrieve based on memory confidence.

        Score > 0.85  ->  memory ONLY (no RAG call)     ~200 tokens
        Score 0.60-0.85 -> memory + 1 RAG chunk         ~600 tokens
        Score < 0.60  ->  3 RAG chunks only              ~2000 tokens

        This routing saves ~70% on API costs for repeated question types.
        """
        memories = await self.search_memories(query)

        best_score = 0.0
        memory_context = ""
        if memories:
            best_score = float(memories[0].get("similarity", 0))
            parts = []
            for m in memories:
                text = f"**{m['concept']}**: {m['summary']}"
                if m.get("fix"):
                    text += f"\nFix: {m['fix']}"
                parts.append(text)
            memory_context = "\n\n".join(parts)

        high = settings.memory_high_confidence
        low = settings.memory_low_confidence

        if best_score > high:
            # High confidence: memory only
            logger.info(
                "Router: memory-only (score=%.2f > %.2f)", best_score, high
            )
            return memory_context

        if best_score >= low:
            # Medium confidence: memory + 1 RAG chunk
            logger.info(
                "Router: memory + 1 RAG chunk (score=%.2f)", best_score
            )
            rag_results = await search_knowledge_base(query, top_k=1)
            rag_context = "\n\n".join(r["content"] for r in rag_results)
            return f"{memory_context}\n\n---\n\n{rag_context}" if rag_context else memory_context

        # Low confidence: full RAG (3 chunks)
        logger.info(
            "Router: full RAG, 3 chunks (score=%.2f < %.2f)", best_score, low
        )
        rag_results = await search_knowledge_base(query, top_k=3)
        return "\n\n---\n\n".join(r["content"] for r in rag_results)


# ---------------------------------------------------------------------------
# Background scheduler
# ---------------------------------------------------------------------------

_agent: MemoryAgent | None = None


def _get_agent() -> MemoryAgent:
    global _agent
    if _agent is None:
        _agent = MemoryAgent()
    return _agent


def _run_compaction() -> None:
    """Synchronous wrapper for the scheduler."""
    agent = _get_agent()
    loop = asyncio.new_event_loop()
    try:
        count = loop.run_until_complete(agent.compact_interactions())
        logger.info("Scheduled compaction complete: %d nuggets", count)
    except Exception:
        logger.exception("Compaction failed")
    finally:
        loop.close()


def start_scheduler() -> None:
    """Start the background compaction scheduler.

    Runs compact_interactions() every 30 minutes.
    """
    interval = settings.memory_compaction_interval_minutes
    schedule.every(interval).minutes.do(_run_compaction)
    logger.info(
        "Memory compaction scheduler started (every %d minutes)", interval
    )
    while True:
        schedule.run_pending()
        # Sleep 10 seconds between checks to avoid busy-waiting
        import time

        time.sleep(10)


async def run_compaction_loop() -> None:
    """Async compaction loop (alternative to schedule-based scheduler)."""
    agent = _get_agent()
    interval = settings.memory_compaction_interval_minutes * 60
    while True:
        try:
            count = await agent.compact_interactions()
            logger.info("Async compaction complete: %d nuggets", count)
        except Exception:
            logger.exception("Compaction failed")
        await asyncio.sleep(interval)
