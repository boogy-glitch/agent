"""LangGraph multi-agent orchestrator.

Defines the stateful workflow that coordinates:
  1. Tweet monitoring & classification
  2. Knowledge retrieval (RAG)
  3. Draft reply generation
  4. Code validation (E2B sandbox)
  5. Human-in-the-loop approval gate
  6. Reply posting
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from config.settings import settings
from database.db import upsert_interaction
from tools.search_docs import search_revenuecat_docs
from tools.validator import validate_code
from tools.x_api import reply_to_tweet

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    tweet_id: str
    tweet_author: str
    tweet_text: str
    docs_context: str
    draft_reply: str
    code_snippet: str
    code_valid: bool
    approved: bool
    messages: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model=settings.claude_model,
    api_key=settings.anthropic_api_key,
    max_tokens=1024,
)


async def classify_tweet(state: AgentState) -> dict:
    """Determine if the tweet is a RevenueCat support question."""
    prompt = (
        "You are a classifier. Decide if this tweet is asking a technical "
        "question about RevenueCat SDKs, in-app purchases, or subscriptions. "
        "Reply ONLY with 'RELEVANT' or 'IRRELEVANT'.\n\n"
        f"Tweet: {state['tweet_text']}"
    )
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    tag = resp.content.strip().upper()
    return {"messages": [AIMessage(content=f"classification={tag}")]}


async def retrieve_docs(state: AgentState) -> dict:
    """Fetch relevant documentation from the knowledge base."""
    results = await search_revenuecat_docs(state["tweet_text"], limit=4)
    context = "\n---\n".join(
        f"[{r.get('section', '')}] {r['content']}" for r in results
    )
    return {"docs_context": context}


async def draft_reply(state: AgentState) -> dict:
    """Generate a helpful reply grounded in the retrieved docs."""
    prompt = (
        "You are a friendly developer advocate for RevenueCat. "
        "Using ONLY the documentation context below, draft a concise, "
        "helpful reply to the tweet. If code is needed, include a short "
        "snippet and mark it with <code>...</code> tags.\n\n"
        f"Documentation:\n{state.get('docs_context', '')}\n\n"
        f"Tweet from @{state['tweet_author']}:\n{state['tweet_text']}"
    )
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    reply_text = resp.content.strip()

    # Extract code snippet if present
    code = ""
    if "<code>" in reply_text and "</code>" in reply_text:
        code = reply_text.split("<code>")[1].split("</code>")[0].strip()

    return {"draft_reply": reply_text, "code_snippet": code}


async def validate_code_node(state: AgentState) -> dict:
    """Run the code snippet in a sandbox to verify correctness."""
    snippet = state.get("code_snippet", "")
    if not snippet:
        return {"code_valid": True}
    result = await validate_code(snippet)
    return {"code_valid": result["success"]}


async def save_interaction(state: AgentState) -> dict:
    """Persist the interaction for HITL review."""
    await upsert_interaction(
        {
            "tweet_id": state["tweet_id"],
            "tweet_author": state["tweet_author"],
            "tweet_text": state["tweet_text"],
            "draft_reply": state["draft_reply"],
            "code_snippet": state.get("code_snippet", ""),
            "code_validated": state.get("code_valid", False),
            "status": "PENDING_APPROVAL",
        }
    )
    return {}


async def post_reply(state: AgentState) -> dict:
    """Post the approved reply to Twitter."""
    if state.get("approved"):
        reply_to_tweet(state["tweet_id"], state["draft_reply"])
    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def is_relevant(state: AgentState) -> str:
    last_msg = state["messages"][-1].content if state["messages"] else ""
    return "retrieve" if "RELEVANT" in last_msg else "skip"


def needs_validation(state: AgentState) -> str:
    return "validate" if state.get("code_snippet") else "save"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_tweet)
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("draft", draft_reply)
    graph.add_node("validate", validate_code_node)
    graph.add_node("save", save_interaction)
    graph.add_node("post", post_reply)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify", is_relevant, {"retrieve": "retrieve", "skip": END}
    )
    graph.add_edge("retrieve", "draft")
    graph.add_conditional_edges(
        "draft", needs_validation, {"validate": "validate", "save": "save"}
    )
    graph.add_edge("validate", "save")
    graph.add_edge("save", "post")
    graph.add_edge("post", END)

    return graph.compile()


workflow = build_graph()
