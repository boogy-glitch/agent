"""Memory Bank page — browse, search, and manage memory nuggets."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Memory Bank | RevenueCat Agent",
    page_icon="RC",
    layout="wide",
)

from config.settings import settings
from database.db import get_supabase

# ---------------------------------------------------------------------------
# Password gate
# ---------------------------------------------------------------------------
if settings.dashboard_password and not st.session_state.get("authenticated"):
    st.warning("Please log in from the main page.")
    st.stop()


st.markdown("# Memory Bank")
st.caption("Compacted knowledge nuggets extracted from successful interactions.")

sb = get_supabase()

# ---------------------------------------------------------------------------
# Search / filter
# ---------------------------------------------------------------------------

search_query = st.text_input(
    "Search by concept or keyword", placeholder="e.g. StoreKit, migration, billing"
)

col_sort, col_min = st.columns(2)
with col_sort:
    sort_by = st.selectbox(
        "Sort by", ["importance", "usage_count", "created_at"], index=0
    )
with col_min:
    min_importance = st.slider("Min importance", 0.0, 1.0, 0.0, 0.05)

# ---------------------------------------------------------------------------
# Fetch nuggets
# ---------------------------------------------------------------------------

query = (
    sb.table("memory_nuggets")
    .select("id, concept, summary, fix, importance, usage_count, created_at")
    .gte("importance", min_importance)
    .order(sort_by, desc=True)
    .limit(100)
)

if search_query:
    query = query.ilike("concept", f"%{search_query}%")

resp = query.execute()
nuggets = resp.data or []

st.markdown(f"**{len(nuggets)}** nuggets found")

# ---------------------------------------------------------------------------
# Importance distribution
# ---------------------------------------------------------------------------

if nuggets:
    import pandas as pd

    df = pd.DataFrame(nuggets)
    if "importance" in df.columns:
        st.markdown("### Importance Distribution")
        hist_data = df["importance"].dropna()
        st.bar_chart(hist_data.value_counts(bins=10).sort_index(), height=200)

# ---------------------------------------------------------------------------
# Nugget table
# ---------------------------------------------------------------------------

st.markdown("### All Nuggets")

for n in nuggets:
    importance = n.get("importance", 0)

    # Color-coded importance bar
    if importance >= 0.8:
        bar_color = "#A6E3A1"  # green
    elif importance >= 0.5:
        bar_color = "#F9E2AF"  # yellow
    else:
        bar_color = "#F38BA8"  # red

    with st.expander(
        f"**{n['concept']}** — importance: {importance:.2f} | "
        f"used {n.get('usage_count', 0)}x"
    ):
        # Importance bar
        pct = int(importance * 100)
        st.markdown(
            f'<div style="background:#313244;border-radius:6px;height:8px;'
            f'margin-bottom:12px">'
            f'<div style="background:{bar_color};width:{pct}%;height:8px;'
            f'border-radius:6px"></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(f"**Summary:** {n.get('summary', '—')}")
        if n.get("fix"):
            st.markdown(f"**Fix:** {n['fix']}")
        st.caption(f"Created: {str(n.get('created_at', ''))[:19]}")

        # Actions
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Delete", key=f"del_{n['id']}", type="secondary"):
                sb.table("memory_nuggets").delete().eq("id", n["id"]).execute()
                st.warning("Deleted.")
                st.rerun()
        with c2:
            new_summary = st.text_area(
                "Edit summary",
                value=n.get("summary", ""),
                key=f"edit_{n['id']}",
                height=80,
            )
            if st.button("Save Edit", key=f"save_{n['id']}"):
                sb.table("memory_nuggets").update(
                    {"summary": new_summary}
                ).eq("id", n["id"]).execute()
                st.success("Updated.")
                st.rerun()
