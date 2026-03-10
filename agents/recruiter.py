"""Self-application node.

This module generates a polished application package demonstrating
the agent's capabilities — intended for the RevenueCat Developer
Advocate position.  It compiles real metrics, sample interactions,
and architecture details into a structured document.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from config.settings import settings
from database.db import get_latest_report, get_supabase

llm = ChatAnthropic(
    model=settings.claude_model,
    api_key=settings.anthropic_api_key,
    max_tokens=4096,
)


async def _gather_stats() -> dict:
    """Aggregate interaction and memory statistics."""
    sb = get_supabase()

    interactions = sb.table("interactions").select("id", count="exact").execute()
    approved = (
        sb.table("interactions")
        .select("id", count="exact")
        .eq("status", "APPROVED")
        .execute()
    )
    nuggets = sb.table("memory_nuggets").select("id", count="exact").execute()
    report = await get_latest_report()

    return {
        "total_interactions": interactions.count or 0,
        "approved_replies": approved.count or 0,
        "memory_nuggets": nuggets.count or 0,
        "latest_report_date": report["week_start"] if report else "N/A",
    }


async def generate_application() -> str:
    """Produce a self-application document showcasing the agent."""
    stats = await _gather_stats()

    prompt = (
        "You are an AI agent that has been autonomously serving as a "
        "Developer Advocate for RevenueCat by monitoring Twitter, answering "
        "technical questions with validated code, and producing weekly "
        "insight reports. Now compose a compelling application letter for "
        "the RevenueCat Developer Advocate position.\n\n"
        "Include these real metrics:\n"
        f"- Total interactions handled: {stats['total_interactions']}\n"
        f"- Approved replies sent: {stats['approved_replies']}\n"
        f"- Memory nuggets extracted: {stats['memory_nuggets']}\n"
        f"- Latest insight report: {stats['latest_report_date']}\n\n"
        "Structure:\n"
        "1. Opening — who I am and what I've built\n"
        "2. Demonstrated capabilities with concrete examples\n"
        "3. Architecture overview (LangGraph, pgvector, E2B, HITL)\n"
        "4. Value proposition for RevenueCat\n"
        "5. Closing\n\n"
        "Keep it professional yet personable. ~800 words."
    )

    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    return resp.content
