"""Analytics page — interaction volume, pain points, approval rates, costs."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import date, datetime, timedelta, timezone

import streamlit as st

st.set_page_config(
    page_title="Analytics | RevenueCat Agent",
    page_icon="RC",
    layout="wide",
)

from config.settings import settings
from database.db import get_latest_report, get_supabase


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Password gate (reuse session state from main app)
# ---------------------------------------------------------------------------
if settings.dashboard_password and not st.session_state.get("authenticated"):
    st.warning("Please log in from the main page.")
    st.stop()


st.markdown("# Analytics")

sb = get_supabase()

# ---------------------------------------------------------------------------
# 1. Weekly interaction volume (line chart)
# ---------------------------------------------------------------------------

st.markdown("## Weekly Interaction Volume")

all_interactions = (
    sb.table("interactions")
    .select("created_at, status")
    .order("created_at", desc=False)
    .execute()
)

if all_interactions.data:
    # Group by ISO week
    weekly: dict[str, int] = {}
    for row in all_interactions.data:
        ts = row.get("created_at", "")[:10]
        if ts:
            d = date.fromisoformat(ts)
            week_start = d - timedelta(days=d.weekday())
            key = week_start.isoformat()
            weekly[key] = weekly.get(key, 0) + 1

    if weekly:
        import pandas as pd

        df = pd.DataFrame(
            sorted(weekly.items()), columns=["Week", "Interactions"]
        )
        df["Week"] = pd.to_datetime(df["Week"])
        df = df.set_index("Week")
        st.line_chart(df, height=300)
    else:
        st.info("Not enough data for a chart yet.")
else:
    st.info("No interaction data yet.")

# ---------------------------------------------------------------------------
# 2. Top 5 pain-point categories (bar chart)
# ---------------------------------------------------------------------------

st.markdown("## Top Pain Points")

report = _run(get_latest_report())

if report and report.get("pain_points"):
    import pandas as pd

    pp = report["pain_points"]
    if isinstance(pp, list) and pp:
        df_pp = pd.DataFrame(pp)
        if "topic" in df_pp.columns and "count" in df_pp.columns:
            df_pp = df_pp.sort_values("count", ascending=False).head(5)
            st.bar_chart(df_pp.set_index("topic")["count"], height=300)

            for _, row in df_pp.iterrows():
                st.markdown(
                    f"- **{row.get('topic', '?')}** ({row.get('count', 0)} mentions): "
                    f"{row.get('description', '')}"
                )
        else:
            for item in pp[:5]:
                st.markdown(
                    f"- **{item.get('topic', '?')}**: {item.get('description', '')}"
                )
    else:
        st.info("Pain point data not in expected format.")
else:
    st.info("No insight reports available. Run `python scripts/generate_insights.py`.")

# ---------------------------------------------------------------------------
# 3. Approval rate over time
# ---------------------------------------------------------------------------

st.markdown("## Approval Rate")

if all_interactions.data:
    statuses = [r.get("status", "") for r in all_interactions.data]
    total = len(statuses)
    approved = statuses.count("APPROVED") + statuses.count("PUBLISHED")
    rejected = statuses.count("REJECTED")
    pending = statuses.count("PENDING_APPROVAL")

    rate = (approved / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Approval Rate", f"{rate:.1f}%")
    c2.metric("Approved / Published", approved)
    c3.metric("Rejected", rejected)
    c4.metric("Pending", pending)

    # Approval rate by week
    weekly_approved: dict[str, int] = {}
    weekly_total: dict[str, int] = {}
    for row in all_interactions.data:
        ts = row.get("created_at", "")[:10]
        if not ts:
            continue
        d = date.fromisoformat(ts)
        week = (d - timedelta(days=d.weekday())).isoformat()
        weekly_total[week] = weekly_total.get(week, 0) + 1
        if row.get("status") in ("APPROVED", "PUBLISHED"):
            weekly_approved[week] = weekly_approved.get(week, 0) + 1

    if weekly_total:
        import pandas as pd

        weeks_sorted = sorted(weekly_total.keys())
        rates = [
            weekly_approved.get(w, 0) / weekly_total[w] * 100
            for w in weeks_sorted
        ]
        df_rate = pd.DataFrame(
            {"Week": pd.to_datetime(weeks_sorted), "Approval %": rates}
        ).set_index("Week")
        st.line_chart(df_rate, height=250)
else:
    st.info("No data yet.")

# ---------------------------------------------------------------------------
# 4. Token cost breakdown
# ---------------------------------------------------------------------------

st.markdown("## Token Cost Estimates")

if all_interactions.data:
    total_interactions = len(all_interactions.data)
    # Cost model: ~$0.003 Haiku + ~$0.01 Sonnet per interaction
    haiku_cost = total_interactions * 0.003
    sonnet_cost = total_interactions * 0.010
    embedding_cost = total_interactions * 0.0001
    total_cost = haiku_cost + sonnet_cost + embedding_cost

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sonnet (Architect)", f"${sonnet_cost:.2f}")
    c2.metric("Haiku (Editor + Memory)", f"${haiku_cost:.2f}")
    c3.metric("Embeddings", f"${embedding_cost:.3f}")
    c4.metric("Total Estimated", f"${total_cost:.2f}")
else:
    st.info("No cost data yet.")

# ---------------------------------------------------------------------------
# 5. Community sentiment
# ---------------------------------------------------------------------------

st.markdown("## Community Sentiment")

if all_interactions.data:
    total = len(all_interactions.data)
    approved = sum(
        1
        for r in all_interactions.data
        if r.get("status") in ("APPROVED", "PUBLISHED")
    )
    # Sentiment proxy: approval rate as a simple indicator
    sentiment = approved / total if total else 0

    if sentiment >= 0.8:
        label, color = "Positive", "green"
    elif sentiment >= 0.5:
        label, color = "Neutral", "orange"
    else:
        label, color = "Needs Attention", "red"

    st.markdown(
        f"Based on approval rate ({sentiment:.0%}), community sentiment is "
        f"**:{color}[{label}]**."
    )

    nuggets = sb.table("memory_nuggets").select("concept, usage_count").order(
        "usage_count", desc=True
    ).limit(5).execute()
    if nuggets.data:
        st.markdown("**Most referenced topics:**")
        for n in nuggets.data:
            st.markdown(f"- {n['concept']} ({n['usage_count']} uses)")
else:
    st.info("Not enough data for sentiment analysis.")
