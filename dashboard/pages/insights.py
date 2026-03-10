"""Product insights page for the Streamlit dashboard."""

from __future__ import annotations

import asyncio

import streamlit as st

from database.db import get_latest_report, get_supabase

st.set_page_config(page_title="Insights | RevenueCat Agent", layout="wide")
st.title("Product Insights")


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# -------------------------------------------------------------------
# Latest weekly report
# -------------------------------------------------------------------

st.header("Latest Weekly Report")

report = _run(get_latest_report())

if report:
    st.markdown(f"**Week of:** {report['week_start']}")
    st.markdown(report["content"])

    st.subheader("Pain Points")
    pain_points = report.get("pain_points", [])
    if pain_points:
        for pp in pain_points:
            st.markdown(
                f"- **{pp.get('topic', 'Unknown')}** "
                f"(mentions: {pp.get('count', '?')}): "
                f"{pp.get('description', '')}"
            )
    else:
        st.info("No pain points extracted this week.")
else:
    st.info("No insight reports generated yet. Run the weekly report script.")

# -------------------------------------------------------------------
# Interaction stats
# -------------------------------------------------------------------

st.header("Interaction Statistics")

sb = get_supabase()

total = sb.table("interactions").select("id", count="exact").execute()
approved = (
    sb.table("interactions")
    .select("id", count="exact")
    .eq("status", "APPROVED")
    .execute()
)
rejected = (
    sb.table("interactions")
    .select("id", count="exact")
    .eq("status", "REJECTED")
    .execute()
)
pending = (
    sb.table("interactions")
    .select("id", count="exact")
    .eq("status", "PENDING_APPROVAL")
    .execute()
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total", total.count or 0)
col2.metric("Approved", approved.count or 0)
col3.metric("Rejected", rejected.count or 0)
col4.metric("Pending", pending.count or 0)

# -------------------------------------------------------------------
# Memory nuggets
# -------------------------------------------------------------------

st.header("Top Memory Nuggets")

nuggets = (
    sb.table("memory_nuggets")
    .select("concept, summary, importance, usage_count")
    .order("importance", desc=True)
    .limit(10)
    .execute()
)

if nuggets.data:
    for n in nuggets.data:
        st.markdown(
            f"- **{n['concept']}** (importance: {n['importance']:.2f}, "
            f"used {n['usage_count']}x): {n['summary']}"
        )
else:
    st.info("No memory nuggets yet. The memory agent will create them.")
