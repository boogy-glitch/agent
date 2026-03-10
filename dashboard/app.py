"""RevenueCat Agent — Control Center.

Main HITL dashboard for reviewing, editing, and approving AI-generated
replies before they are posted to Twitter/X.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

import streamlit as st

st.set_page_config(
    page_title="RevenueCat Agent — Control Center",
    page_icon="RC",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config.settings import settings
from database.db import (
    get_pending_interactions,
    get_supabase,
    update_interaction_status,
    upsert_interaction,
)
from tools.x_api import reply_to_tweet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine from Streamlit's sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _detect_platform(text: str) -> str:
    """Guess the platform tag from tweet text."""
    t = text.lower()
    if any(w in t for w in ("swift", "storekit", "ios", "xcode", "skerror")):
        return "iOS"
    if any(w in t for w in ("kotlin", "android", "billing", "play store")):
        return "Android"
    if any(w in t for w in ("flutter", "dart")):
        return "Flutter"
    if any(w in t for w in ("react native", "react-native", "rn ")):
        return "React Native"
    if any(w in t for w in ("unity", "c#")):
        return "Unity"
    return "General"


def _platform_color(platform: str) -> str:
    colors = {
        "iOS": "#007AFF",
        "Android": "#3DDC84",
        "Flutter": "#027DFD",
        "React Native": "#61DAFB",
        "Unity": "#222C37",
        "General": "#6B7280",
    }
    return colors.get(platform, "#6B7280")


def _detect_language(code: str) -> str:
    """Guess code language for syntax highlighting."""
    if "func " in code or "import StoreKit" in code or "Purchases.shared" in code:
        return "swift"
    if "fun " in code or "Purchases.sharedInstance" in code:
        return "kotlin"
    if "await " in code and ("Purchases." in code or "import 'package:" in code):
        return "dart"
    if "import " in code and ("React" in code or "require(" in code):
        return "javascript"
    return "python"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ~ 4 chars."""
    return len(text) // 4


def _validation_badge(validated: bool | None) -> str:
    if validated is True:
        return "Validated"
    elif validated is False:
        return "Failed"
    return "Not checked"


# ---------------------------------------------------------------------------
# Password protection
# ---------------------------------------------------------------------------


def check_password() -> bool:
    """Return True if the user has entered the correct password."""
    pwd = settings.dashboard_password
    if not pwd:
        return True  # no password set — allow access

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("### Login")
    entered = st.text_input("Password", type="password", key="pw_input")
    if st.button("Sign in", key="pw_btn"):
        if entered == pwd:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .metric-card {
        background: #1E1E2E;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #313244;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #CDD6F4;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #A6ADC8;
        margin-top: 4px;
    }
    .platform-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
    }
    .tweet-card {
        background: #1E1E2E;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #313244;
        margin-bottom: 16px;
    }
    .divider {
        border-top: 1px solid #313244;
        margin: 16px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Agent Status")

    # Agent state (stored in session_state for UI toggle)
    if "agent_running" not in st.session_state:
        st.session_state.agent_running = True

    if st.session_state.agent_running:
        st.success("RUNNING")
        if st.button("Pause Agent", key="pause_btn"):
            st.session_state.agent_running = False
            st.rerun()
    else:
        st.warning("PAUSED")
        if st.button("Start Agent", key="start_btn"):
            st.session_state.agent_running = True
            st.rerun()

    st.markdown("---")

    now = datetime.now(timezone.utc)
    st.markdown(f"**Last scan:** {now.strftime('%H:%M:%S UTC')}")

    scan_interval = settings.memory_compaction_interval_minutes
    st.markdown(f"**Next scan:** in {scan_interval} min")

    st.markdown("---")

    # Quick stats
    sb = get_supabase()
    total_resp = sb.table("interactions").select("id", count="exact").execute()
    nuggets_resp = sb.table("memory_nuggets").select("id", count="exact").execute()
    total_count = total_resp.count or 0
    nuggets_count = nuggets_resp.count or 0

    # Rough cost estimate: ~$0.003 per interaction (Haiku) + ~$0.01 (Sonnet)
    est_cost = total_count * 0.013
    st.markdown(f"**Total interactions:** {total_count}")
    st.markdown(f"**Memory nuggets:** {nuggets_count}")
    st.markdown(f"**Est. API costs:** ${est_cost:.2f}")

    st.markdown("---")
    st.caption("RevenueCat AI Developer Advocate")


# ---------------------------------------------------------------------------
# Header metrics
# ---------------------------------------------------------------------------

st.markdown("# RevenueCat Agent — Control Center")

sb = get_supabase()

pending_resp = (
    sb.table("interactions")
    .select("id", count="exact")
    .eq("status", "PENDING_APPROVAL")
    .execute()
)
approved_resp = (
    sb.table("interactions")
    .select("id", count="exact")
    .eq("status", "APPROVED")
    .execute()
)
rejected_resp = (
    sb.table("interactions")
    .select("id", count="exact")
    .eq("status", "REJECTED")
    .execute()
)
published_resp = (
    sb.table("interactions")
    .select("id", count="exact")
    .eq("status", "PUBLISHED")
    .execute()
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{pending_resp.count or 0}</div>'
        f'<div class="metric-label">Pending</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{approved_resp.count or 0}</div>'
        f'<div class="metric-label">Approved</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{rejected_resp.count or 0}</div>'
        f'<div class="metric-label">Rejected</div></div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{published_resp.count or 0}</div>'
        f'<div class="metric-label">Live on X</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ---------------------------------------------------------------------------
# Pending interaction queue
# ---------------------------------------------------------------------------

st.markdown("## Approval Queue")

interactions = _run(get_pending_interactions())

if not interactions:
    st.info("No pending interactions. The agent is monitoring Twitter.")
else:
    for ix in interactions:
        tweet_text = ix.get("tweet_text", "")
        tweet_author = ix.get("tweet_author", "unknown")
        draft = ix.get("draft_reply", "")
        code = ix.get("code_snippet", "")
        validated = ix.get("code_validated")
        created = ix.get("created_at", "")
        platform = _detect_platform(tweet_text)
        plat_color = _platform_color(platform)

        st.markdown(f'<div class="tweet-card">', unsafe_allow_html=True)

        left, right = st.columns([0.4, 0.6])

        # --- Left column: original tweet ---
        with left:
            st.markdown(f"### @{tweet_author}")
            st.markdown(
                f'<span class="platform-tag" style="background:{plat_color}">'
                f"{platform}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"> {tweet_text}")
            if created:
                ts = created if isinstance(created, str) else str(created)
                st.caption(f"Received: {ts[:19]}")

            # Token cost estimate
            total_tokens = _estimate_tokens(tweet_text) + _estimate_tokens(draft)
            if code:
                total_tokens += _estimate_tokens(code)
            st.caption(f"Est. tokens: ~{total_tokens}")

        # --- Right column: AI reply ---
        with right:
            edited_reply = st.text_area(
                "AI Reply (editable)",
                value=draft,
                height=120,
                key=f"reply_{ix['id']}",
            )

            if code:
                lang = _detect_language(code)
                st.code(code, language=lang)

            # Validation status
            badge = _validation_badge(validated)
            if validated is True:
                st.success(f"E2B: {badge}")
            elif validated is False:
                st.error(f"E2B: {badge}")
            else:
                st.warning(f"E2B: {badge}")

            # Source info
            if ix.get("compacted"):
                st.caption("Source: Memory nugget")
            else:
                st.caption("Source: RAG documentation")

        # --- Action buttons ---
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)

        with b1:
            if st.button("Approve & Post", key=f"approve_{ix['id']}", type="primary"):
                try:
                    reply_to_tweet(ix["tweet_id"], edited_reply)
                    _run(update_interaction_status(ix["id"], "PUBLISHED"))
                    st.success("Posted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to post: {e}")

        with b2:
            if st.button("Edit & Approve", key=f"edit_approve_{ix['id']}"):
                if edited_reply != draft:
                    _run(
                        upsert_interaction(
                            {
                                "id": ix["id"],
                                "tweet_id": ix["tweet_id"],
                                "tweet_author": tweet_author,
                                "tweet_text": tweet_text,
                                "draft_reply": edited_reply,
                                "code_snippet": code,
                                "code_validated": validated,
                                "status": "PENDING_APPROVAL",
                            }
                        )
                    )
                    try:
                        reply_to_tweet(ix["tweet_id"], edited_reply)
                        _run(update_interaction_status(ix["id"], "PUBLISHED"))
                        st.success("Edited reply posted!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to post: {e}")
                else:
                    st.warning("No edits detected. Use 'Approve & Post' instead.")

        with b3:
            if st.button("Reject", key=f"reject_{ix['id']}"):
                _run(update_interaction_status(ix["id"], "REJECTED"))
                st.warning("Rejected.")
                st.rerun()

        with b4:
            if st.button("Regenerate", key=f"regen_{ix['id']}"):
                _run(update_interaction_status(ix["id"], "REGENERATING"))
                st.info("Queued for regeneration. The agent will re-draft.")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")
