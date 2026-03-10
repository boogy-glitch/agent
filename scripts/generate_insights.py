"""Generate a weekly insight report from recent interactions.

Analyses all interactions from the past 7 days, identifies recurring
pain points and themes, and stores the report in the database.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from config.settings import settings
from database.db import get_supabase, insert_insight_report

llm = ChatAnthropic(
    model=settings.claude_model,
    api_key=settings.anthropic_api_key,
    max_tokens=4096,
)


async def generate_report() -> str:
    """Build and store the weekly insight report. Returns the report id."""
    sb = get_supabase()
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    resp = (
        sb.table("interactions")
        .select("tweet_text, draft_reply, code_snippet, status")
        .gte("created_at", since)
        .execute()
    )
    interactions = resp.data
    if not interactions:
        print("No interactions in the past week.")
        return ""

    block = "\n---\n".join(
        f"Q: {i['tweet_text']}\nA: {i['draft_reply']}\nStatus: {i['status']}"
        for i in interactions
    )

    prompt = (
        "You are a product-insights analyst for RevenueCat. Analyse the "
        "following developer support interactions from the past week and "
        "produce:\n\n"
        "1. An executive summary (2-3 paragraphs)\n"
        "2. A JSON array of pain_points, each with keys: topic, count, "
        "description\n\n"
        "Return the summary first, then on a new line the JSON array "
        "(no markdown fences).\n\n"
        f"Interactions:\n{block}"
    )

    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    content = resp.content.strip()

    # Try to extract the JSON pain-points array from the end of the response
    pain_points: list[dict] = []
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("["):
            try:
                pain_points = json.loads("\n".join(lines[i:]))
                content = "\n".join(lines[:i]).strip()
            except json.JSONDecodeError:
                pass
            break

    week_start = date.today() - timedelta(days=date.today().weekday())
    report_id = await insert_insight_report(week_start, content, pain_points)

    print(f"Report {report_id} saved for week of {week_start}")
    return report_id


if __name__ == "__main__":
    asyncio.run(generate_report())
