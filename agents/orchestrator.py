"""LangGraph multi-agent orchestrator.

Full pipeline: scout -> architect -> validator -> editor -> END
with retry logic on validation failure (up to 2 retries).

All static system prompts use Anthropic prompt caching to cut costs ~90%.
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

from anthropic import Anthropic
from langgraph.graph import END, StateGraph

from agents.memory_agent import MemoryAgent
from config.settings import settings
from database.db import get_supabase, upsert_interaction
from tools.validator import validate_code
from tools.x_api import search_tweets

logger = logging.getLogger("orchestrator")

# ---------------------------------------------------------------------------
# Anthropic client (direct API for prompt caching support)
# ---------------------------------------------------------------------------

_anthropic = Anthropic(api_key=settings.anthropic_api_key)
_memory_agent = MemoryAgent(anthropic_client=_anthropic)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    tweet_id: str
    tweet_text: str
    tweet_author: str
    tweet_url: str
    memory_context: str
    rag_context: str
    draft_reply: str
    code_snippet: str
    code_validated: bool
    edited_reply: str
    status: str
    error_message: str
    # Internal: retry tracking
    _validation_attempts: int


# ---------------------------------------------------------------------------
# Cached system prompts
# ---------------------------------------------------------------------------

# Static prompts are marked with cache_control so the Anthropic API caches
# them across calls, saving ~90% on input token costs for repeated use.

_ARCHITECT_SYSTEM: list[dict[str, Any]] = [
    {
        "type": "text",
        "text": (
            "You are a Senior RevenueCat SDK expert and developer advocate.\n\n"
            "Rules:\n"
            "- Answer ONLY with technical solutions using RevenueCat SDKs.\n"
            "- ALWAYS include a code snippet in Swift, Kotlin, or Dart "
            "(pick the language that best matches the question).\n"
            "- Wrap code in <code lang=\"LANG\">...</code> tags.\n"
            "- Be concise — the reply will be posted on Twitter.\n"
            "- Ground your answer in the provided context. "
            "Do not hallucinate API methods.\n"
        ),
        "cache_control": {"type": "ephemeral"},
    }
]

_EDITOR_SYSTEM: list[dict[str, Any]] = [
    {
        "type": "text",
        "text": (
            "You are a grumpy senior developer who hates corporate AI tone.\n"
            "Rewrite this reply to sound like a real developer wrote it.\n\n"
            "Rules:\n"
            "- Max 280 characters for the tweet text (code can be in a follow-up).\n"
            "- NEVER use these words: 'delve', 'unlock', 'comprehensive', 'leverage'.\n"
            "- Start with empathy for the developer's pain.\n"
            "- End with the concrete solution.\n"
            "- Keep any <code>...</code> block EXACTLY as-is.\n"
            "- Sound human. Be direct. No fluff.\n"
        ),
        "cache_control": {"type": "ephemeral"},
    }
]

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def scout_node(state: AgentState) -> dict:
    """Search Twitter for RevenueCat-related technical questions.

    Filters: last 24h, min 10 followers, not already in the DB.
    """
    logger.info("Scout: searching for tweets...")
    tweets = search_tweets()

    if not tweets:
        logger.info("Scout: no matching tweets found")
        return {"status": "NO_TWEETS", "error_message": "No matching tweets found"}

    # Check which tweet_ids we've already handled
    sb = get_supabase()
    existing_resp = sb.table("interactions").select("tweet_id").execute()
    existing_ids = {r["tweet_id"] for r in existing_resp.data} if existing_resp.data else set()

    for tweet in tweets:
        if tweet["id"] not in existing_ids:
            logger.info(
                "Scout: found tweet %s from @%s",
                tweet["id"],
                tweet["author_username"],
            )
            return {
                "tweet_id": tweet["id"],
                "tweet_text": tweet["text"],
                "tweet_author": tweet["author_username"],
                "tweet_url": tweet["url"],
                "status": "SCOUTED",
                "_validation_attempts": 0,
            }

    logger.info("Scout: all found tweets already handled")
    return {"status": "NO_NEW_TWEETS", "error_message": "All tweets already handled"}


async def architect_node(state: AgentState) -> dict:
    """Generate a technically grounded reply with code snippet.

    Uses the Token Efficiency Router from MemoryAgent to decide
    how much context to fetch (memory-only, partial RAG, or full RAG).
    """
    tweet_text = state.get("tweet_text", "")
    tweet_author = state.get("tweet_author", "")

    logger.info("Architect: generating reply for @%s", tweet_author)

    # Get optimised context via the memory agent's router
    context = await _memory_agent.get_context_for_query(tweet_text)

    # Build the system prompt with context injected (cached static part +
    # dynamic context part)
    system_blocks = _ARCHITECT_SYSTEM + [
        {
            "type": "text",
            "text": f"Context:\n{context}" if context else "No context available.",
        }
    ]

    response = _anthropic.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=system_blocks,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Tweet from @{tweet_author}:\n\n{tweet_text}\n\n"
                    "Write a helpful reply with a code snippet."
                ),
            }
        ],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    reply_text = response.content[0].text.strip()

    # Extract code snippet from <code> tags
    code_snippet = ""
    code_match = re.search(
        r'<code(?:\s+lang="[^"]*")?>(.*?)</code>', reply_text, re.DOTALL
    )
    if code_match:
        code_snippet = code_match.group(1).strip()

    return {
        "draft_reply": reply_text,
        "code_snippet": code_snippet,
        "memory_context": context,
        "status": "DRAFTED",
    }


async def validator_node(state: AgentState) -> dict:
    """Validate the code snippet in an E2B sandbox.

    Tracks attempts so the router can retry architect up to 2 times.
    """
    snippet = state.get("code_snippet", "")
    attempts = state.get("_validation_attempts", 0) + 1

    if not snippet:
        logger.info("Validator: no code snippet, skipping validation")
        return {"code_validated": True, "_validation_attempts": attempts}

    logger.info("Validator: checking code (attempt %d)", attempts)
    result = await validate_code(snippet)

    if result["success"]:
        logger.info("Validator: code passed")
        return {"code_validated": True, "_validation_attempts": attempts}

    logger.warning("Validator: code failed — %s", result["error"])
    return {
        "code_validated": False,
        "_validation_attempts": attempts,
        "error_message": result["error"],
    }


async def editor_node(state: AgentState) -> dict:
    """Rewrite the reply in an authentic developer voice.

    Uses Claude Haiku for cost efficiency. The static system prompt
    is cached via prompt caching.
    """
    draft = state.get("draft_reply", "")
    tweet_author = state.get("tweet_author", "")

    logger.info("Editor: polishing reply for @%s", tweet_author)

    response = _anthropic.messages.create(
        model=settings.claude_model_haiku,
        max_tokens=512,
        system=_EDITOR_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Original draft reply to @{tweet_author}:\n\n{draft}\n\n"
                    "Rewrite it."
                ),
            }
        ],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    edited = response.content[0].text.strip()

    # Persist the interaction for HITL review
    await upsert_interaction(
        {
            "tweet_id": state.get("tweet_id", ""),
            "tweet_author": tweet_author,
            "tweet_text": state.get("tweet_text", ""),
            "draft_reply": edited,
            "code_snippet": state.get("code_snippet", ""),
            "code_validated": state.get("code_validated", False),
            "status": "PENDING_APPROVAL",
        }
    )

    logger.info("Editor: done — interaction saved as PENDING_APPROVAL")
    return {"edited_reply": edited, "status": "PENDING_APPROVAL"}


async def fail_node(state: AgentState) -> dict:
    """Terminal node when validation fails after max retries."""
    logger.error(
        "Pipeline FAILED for tweet %s after %d validation attempts: %s",
        state.get("tweet_id", "?"),
        state.get("_validation_attempts", 0),
        state.get("error_message", "unknown"),
    )
    await upsert_interaction(
        {
            "tweet_id": state.get("tweet_id", ""),
            "tweet_author": state.get("tweet_author", ""),
            "tweet_text": state.get("tweet_text", ""),
            "draft_reply": state.get("draft_reply", ""),
            "code_snippet": state.get("code_snippet", ""),
            "code_validated": False,
            "status": "FAILED",
        }
    )
    return {"status": "FAILED"}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

MAX_VALIDATION_RETRIES = 2


def route_after_scout(state: AgentState) -> str:
    """Only proceed to architect if we found a tweet."""
    status = state.get("status", "")
    if status == "SCOUTED":
        return "architect"
    return "end"


def route_after_validator(state: AgentState) -> str:
    """After validation: pass -> editor, fail -> retry or give up."""
    if state.get("code_validated"):
        return "editor"
    attempts = state.get("_validation_attempts", 0)
    if attempts < MAX_VALIDATION_RETRIES:
        return "retry_architect"
    return "fail"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("scout", scout_node)
    graph.add_node("architect", architect_node)
    graph.add_node("validator", validator_node)
    graph.add_node("editor", editor_node)
    graph.add_node("retry_architect", architect_node)  # same fn, different node name
    graph.add_node("fail", fail_node)

    # Edges
    graph.set_entry_point("scout")
    graph.add_conditional_edges(
        "scout",
        route_after_scout,
        {"architect": "architect", "end": END},
    )
    graph.add_edge("architect", "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "editor": "editor",
            "retry_architect": "retry_architect",
            "fail": "fail",
        },
    )
    # After a retry, go back through validation
    graph.add_edge("retry_architect", "validator")
    graph.add_edge("editor", END)
    graph.add_edge("fail", END)

    return graph.compile()


workflow = build_graph()
