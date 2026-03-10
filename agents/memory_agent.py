"""Always-On Memory compaction agent.

Periodically scans recent interactions, extracts recurring patterns and
pain-points, and compacts them into reusable *memory nuggets* stored with
pgvector embeddings for fast retrieval.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from config.settings import settings
from database.db import (
    get_supabase,
    upsert_memory_nugget,
)
from tools.search_docs import _embed

llm = ChatAnthropic(
    model=settings.claude_model,
    api_key=settings.anthropic_api_key,
    max_tokens=1024,
)


async def _fetch_recent_interactions(hours: int = 24) -> list[dict]:
    """Pull interactions from the last *hours* hours."""
    sb = get_supabase()
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    resp = (
        sb.table("interactions")
        .select("tweet_text, draft_reply, code_snippet")
        .gte("created_at", since)
        .execute()
    )
    return resp.data


async def compact_memories() -> int:
    """Run one compaction cycle and return the number of new nuggets created."""
    interactions = await _fetch_recent_interactions(
        hours=settings.memory_compaction_interval_hours
    )
    if not interactions:
        return 0

    # Build a summary prompt
    block = "\n---\n".join(
        f"Q: {i['tweet_text']}\nA: {i['draft_reply']}" for i in interactions
    )
    prompt = (
        "You are an expert at extracting reusable developer-support knowledge. "
        "Analyse the following Q&A pairs from RevenueCat support interactions "
        "and extract distinct *memory nuggets*. Each nugget must have:\n"
        "  - concept: a short label (e.g. 'StoreKit 2 migration')\n"
        "  - summary: one-paragraph explanation of the issue\n"
        "  - fix: the recommended solution or code pattern\n"
        "  - importance: a float 0-1 estimating how common the issue is\n\n"
        "Return valid JSON: a list of objects with those four keys.\n\n"
        f"Interactions:\n{block}"
    )

    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    import json

    try:
        nuggets = json.loads(resp.content)
    except json.JSONDecodeError:
        return 0

    created = 0
    for n in nuggets:
        embedding = _embed(f"{n['concept']} {n['summary']}")
        await upsert_memory_nugget(
            {
                "concept": n["concept"],
                "summary": n["summary"],
                "fix": n.get("fix", ""),
                "embedding": embedding,
                "importance": float(n.get("importance", 0.5)),
            }
        )
        created += 1
    return created


async def run_compaction_loop() -> None:
    """Run the memory compaction loop indefinitely."""
    interval = settings.memory_compaction_interval_hours * 3600
    while True:
        count = await compact_memories()
        print(f"[memory-agent] Compacted {count} nuggets")
        await asyncio.sleep(interval)
