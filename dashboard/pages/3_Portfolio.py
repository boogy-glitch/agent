"""Portfolio page — public-facing showcase for RevenueCat."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Portfolio | RevenueCat Agent",
    page_icon="RC",
    layout="wide",
)

from config.settings import settings, VERSION
from database.db import get_latest_report, get_supabase

GITHUB_URL = "https://github.com/boogy-glitch/agent"
PORTFOLIO_MD = Path(__file__).resolve().parent.parent / "portfolio.md"


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# No password gate — this page is the public portfolio

st.markdown("# RevenueCat AI Developer Advocate")
st.markdown(f"*Agent v{VERSION} — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}*")

st.markdown("---")

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

st.markdown("## Architecture")

st.code(
    """
Twitter/X Scout (9 keywords, 24h window)
       |
       v
LangGraph Orchestrator
  |-- Architect Node (Claude Sonnet + prompt caching)
  |     |-- Memory Agent (Token Efficiency Router)
  |     |     |-- Score > 0.85: memory only (~200 tokens)
  |     |     |-- Score 0.60-0.85: memory + 1 RAG chunk
  |     |     |-- Score < 0.60: 3 RAG chunks (~2000 tokens)
  |     |-- pgvector Knowledge Base (Voyage-3 embeddings)
  |-- Validator Node (E2B sandbox + static analysis)
  |-- Editor Node (Claude Haiku, authentic dev voice)
  |-- HITL Dashboard (Streamlit, password-protected)
       |
       v
Post to Twitter/X (human-approved only)
""",
    language=None,
)

# ---------------------------------------------------------------------------
# Live metrics
# ---------------------------------------------------------------------------

st.markdown("## Live Metrics")

sb = get_supabase()

total = sb.table("interactions").select("id", count="exact").execute()
approved = (
    sb.table("interactions")
    .select("id", count="exact")
    .eq("status", "APPROVED")
    .execute()
)
published = (
    sb.table("interactions")
    .select("id", count="exact")
    .eq("status", "PUBLISHED")
    .execute()
)
nuggets = sb.table("memory_nuggets").select("id", count="exact").execute()

validated = (
    sb.table("interactions")
    .select("id", count="exact")
    .eq("code_validated", True)
    .execute()
)
with_code = (
    sb.table("interactions")
    .select("id", count="exact")
    .neq("code_snippet", "")
    .execute()
)
val_rate = (
    f"{validated.count / with_code.count * 100:.0f}%"
    if with_code.count
    else "100%"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Interactions", total.count or 0)
c2.metric("Approved", approved.count or 0)
c3.metric("Published", published.count or 0)
c4.metric("Validation Rate", val_rate)
c5.metric("Memory Nuggets", nuggets.count or 0)

# ---------------------------------------------------------------------------
# Memory Bank (live)
# ---------------------------------------------------------------------------

st.markdown("## Memory Bank (Live)")
st.caption("The agent learns from every interaction. These are its distilled insights.")

nugget_data = (
    sb.table("memory_nuggets")
    .select("concept, summary, fix, importance, usage_count")
    .order("importance", desc=True)
    .limit(10)
    .execute()
)

if nugget_data.data:
    for n in nugget_data.data:
        imp = n.get("importance", 0)
        bar_pct = int(imp * 100)
        st.markdown(
            f"**{n['concept']}** — importance {imp:.2f} | used {n.get('usage_count', 0)}x"
        )
        st.progress(bar_pct / 100)
        st.caption(n.get("summary", ""))
        if n.get("fix"):
            st.code(n["fix"][:200], language=None)
        st.markdown("")
else:
    st.info("Memory bank will populate as the agent handles interactions.")

# ---------------------------------------------------------------------------
# Sample interactions
# ---------------------------------------------------------------------------

st.markdown("## Recent Interactions")

samples = (
    sb.table("interactions")
    .select("tweet_author, tweet_text, draft_reply, code_snippet, code_validated, status")
    .in_("status", ["APPROVED", "PUBLISHED"])
    .order("created_at", desc=True)
    .limit(5)
    .execute()
)

if samples.data:
    for s in samples.data:
        with st.expander(f"@{s.get('tweet_author', '?')} — {s.get('status', '')}"):
            st.markdown(f"> {s.get('tweet_text', '')}")
            st.markdown(f"**Reply:** {s.get('draft_reply', '')}")
            if s.get("code_snippet"):
                st.code(s["code_snippet"])
            badge = "Validated" if s.get("code_validated") else "Not validated"
            st.caption(f"Code: {badge}")
else:
    st.info("No approved interactions yet.")

# ---------------------------------------------------------------------------
# Weekly report
# ---------------------------------------------------------------------------

st.markdown("## Latest Weekly Report")

report = _run(get_latest_report())
if report:
    st.markdown(f"*Week of {report['week_start']}*")
    st.markdown(report["content"][:2000])
else:
    st.info("No weekly reports generated yet.")

# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

st.markdown("## Links")
st.markdown(f"- **GitHub:** [{GITHUB_URL}]({GITHUB_URL})")
st.markdown(f"- **Agent version:** {VERSION}")
st.markdown("- **Stack:** LangGraph + pgvector + Claude Sonnet + E2B + Streamlit")

st.markdown("---")
st.caption("Built as a proof-of-concept AI Developer Advocate for RevenueCat.")
