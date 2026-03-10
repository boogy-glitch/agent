"""Streamlit Human-in-the-Loop dashboard."""

from __future__ import annotations

import asyncio

import streamlit as st

from database.db import (
    get_pending_interactions,
    update_interaction_status,
)
from tools.x_api import reply_to_tweet

st.set_page_config(page_title="RevenueCat Agent Dashboard", layout="wide")
st.title("RevenueCat AI Developer Advocate")


def _run(coro):
    """Helper to run async code from Streamlit's sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# -------------------------------------------------------------------
# Pending interactions
# -------------------------------------------------------------------

st.header("Pending Approvals")

interactions = _run(get_pending_interactions())

if not interactions:
    st.info("No pending interactions. The agent is monitoring Twitter.")
else:
    for ix in interactions:
        with st.expander(f"@{ix['tweet_author']} — {ix['tweet_text'][:80]}..."):
            st.markdown(f"**Tweet:** {ix['tweet_text']}")
            st.markdown(f"**Draft reply:** {ix['draft_reply']}")
            if ix.get("code_snippet"):
                st.code(ix["code_snippet"], language="python")
                validated = ix.get("code_validated", False)
                st.markdown(
                    f"Code validated: {'Yes' if validated else 'No'}"
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Approve & Send", key=f"approve_{ix['id']}"):
                    reply_to_tweet(ix["tweet_id"], ix["draft_reply"])
                    _run(update_interaction_status(ix["id"], "APPROVED"))
                    st.success("Reply sent!")
                    st.rerun()
            with col2:
                if st.button("Reject", key=f"reject_{ix['id']}"):
                    _run(update_interaction_status(ix["id"], "REJECTED"))
                    st.warning("Interaction rejected.")
                    st.rerun()
            with col3:
                edited = st.text_area(
                    "Edit reply", value=ix["draft_reply"], key=f"edit_{ix['id']}"
                )
                if st.button("Send Edited", key=f"send_edit_{ix['id']}"):
                    reply_to_tweet(ix["tweet_id"], edited)
                    _run(update_interaction_status(ix["id"], "APPROVED"))
                    st.success("Edited reply sent!")
                    st.rerun()
