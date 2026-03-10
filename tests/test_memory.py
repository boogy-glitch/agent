"""Tests for the Always-On Memory compaction system.

All external dependencies (DB, LLM, embeddings) are mocked so these
tests run without network access or API keys.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# We need to mock settings before importing the agent so it doesn't
# try to read env vars at import time.
import sys
from types import SimpleNamespace

_mock_settings = SimpleNamespace(
    anthropic_api_key="test-key",
    claude_model="claude-sonnet-4-20250514",
    claude_model_haiku="claude-haiku-4-5-20251001",
    embedding_model="voyage-3",
    embedding_model_fallback="text-embedding-3-small",
    embedding_dimensions=1536,
    similarity_threshold=0.78,
    memory_compaction_min_interactions=5,
    memory_compaction_interval_minutes=30,
    memory_compaction_interval_hours=6,
    memory_high_confidence=0.85,
    memory_low_confidence=0.60,
    voyage_api_key="test-voyage-key",
    supabase_url="https://test.supabase.co",
    supabase_key="test-key",
)

# Patch settings before any project imports
with patch.dict("sys.modules", {}):
    pass

import importlib

# Patch config.settings at the module level
_settings_mod = MagicMock()
_settings_mod.settings = _mock_settings
sys.modules["config.settings"] = _settings_mod
sys.modules["config"] = MagicMock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1] * 1536


def _make_interaction(i: int) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "tweet_id": f"tweet_{i}",
        "tweet_author": f"user_{i}",
        "tweet_text": f"How do I implement subscriptions in iOS? (variant {i})",
        "draft_reply": f"Use Purchases.shared.purchase(package:) to start a purchase flow. (variant {i})",
        "code_snippet": "Purchases.shared.purchase(package: pkg) { ... }",
        "code_validated": True,
        "status": "APPROVED",
        "compacted": False,
    }


def _make_nugget_response() -> str:
    return json.dumps(
        [
            {
                "concept": "iOS subscription purchase",
                "summary": "Developers frequently ask how to implement subscription purchases on iOS using RevenueCat.",
                "fix": "Use Purchases.shared.purchase(package:) with a completion handler.",
                "importance": 0.8,
            }
        ]
    )


def _make_memory_row(similarity: float) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "concept": "iOS subscription purchase",
        "summary": "Use Purchases.shared.purchase(package:) for iOS subscriptions.",
        "fix": "Purchases.shared.purchase(package: pkg) { txn, info, err, cancelled in ... }",
        "importance": 0.8,
        "usage_count": 3,
        "similarity": similarity,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompactInteractions:
    """Tests for MemoryAgent.compact_interactions()."""

    @pytest.mark.asyncio
    @patch("agents.memory_agent.mark_interactions_compacted", new_callable=AsyncMock)
    @patch("agents.memory_agent.upsert_memory_nugget", new_callable=AsyncMock)
    @patch("agents.memory_agent.embed", return_value=FAKE_EMBEDDING)
    @patch("agents.memory_agent.get_uncompacted_interactions", new_callable=AsyncMock)
    async def test_compacts_when_enough_interactions(
        self, mock_get, mock_embed, mock_upsert, mock_mark
    ):
        """Should create nuggets when >= 5 uncompacted interactions exist."""
        mock_get.return_value = [_make_interaction(i) for i in range(6)]

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=_make_nugget_response())]
        mock_client.messages.create.return_value = mock_response

        from agents.memory_agent import MemoryAgent

        agent = MemoryAgent(anthropic_client=mock_client)
        count = await agent.compact_interactions()

        assert count == 1
        mock_upsert.assert_called_once()
        # Should mark exactly 5 interactions as compacted
        mock_mark.assert_called_once()
        marked_ids = mock_mark.call_args[0][0]
        assert len(marked_ids) == 5

        # Verify Haiku model was used
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    @patch("agents.memory_agent.get_uncompacted_interactions", new_callable=AsyncMock)
    async def test_skips_when_too_few_interactions(self, mock_get):
        """Should skip compaction when fewer than 5 interactions exist."""
        mock_get.return_value = [_make_interaction(i) for i in range(3)]

        from agents.memory_agent import MemoryAgent

        agent = MemoryAgent(anthropic_client=MagicMock())
        count = await agent.compact_interactions()

        assert count == 0

    @pytest.mark.asyncio
    @patch("agents.memory_agent.mark_interactions_compacted", new_callable=AsyncMock)
    @patch("agents.memory_agent.upsert_memory_nugget", new_callable=AsyncMock)
    @patch("agents.memory_agent.embed", return_value=FAKE_EMBEDDING)
    @patch("agents.memory_agent.get_uncompacted_interactions", new_callable=AsyncMock)
    async def test_handles_malformed_llm_response(
        self, mock_get, mock_embed, mock_upsert, mock_mark
    ):
        """Should return 0 nuggets if the LLM returns invalid JSON."""
        mock_get.return_value = [_make_interaction(i) for i in range(5)]

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not valid json at all")]
        mock_client.messages.create.return_value = mock_response

        from agents.memory_agent import MemoryAgent

        agent = MemoryAgent(anthropic_client=mock_client)
        count = await agent.compact_interactions()

        assert count == 0
        mock_upsert.assert_not_called()

    @pytest.mark.asyncio
    @patch("agents.memory_agent.mark_interactions_compacted", new_callable=AsyncMock)
    @patch("agents.memory_agent.upsert_memory_nugget", new_callable=AsyncMock)
    @patch("agents.memory_agent.embed", return_value=FAKE_EMBEDDING)
    @patch("agents.memory_agent.get_uncompacted_interactions", new_callable=AsyncMock)
    async def test_handles_single_object_response(
        self, mock_get, mock_embed, mock_upsert, mock_mark
    ):
        """Should handle LLM returning a single object instead of an array."""
        mock_get.return_value = [_make_interaction(i) for i in range(5)]

        single = json.dumps(
            {
                "concept": "test concept",
                "summary": "test summary",
                "fix": "test fix",
                "importance": 0.7,
            }
        )
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=single)]
        mock_client.messages.create.return_value = mock_response

        from agents.memory_agent import MemoryAgent

        agent = MemoryAgent(anthropic_client=mock_client)
        count = await agent.compact_interactions()

        assert count == 1

    @pytest.mark.asyncio
    @patch("agents.memory_agent.mark_interactions_compacted", new_callable=AsyncMock)
    @patch("agents.memory_agent.upsert_memory_nugget", new_callable=AsyncMock)
    @patch("agents.memory_agent.embed", return_value=FAKE_EMBEDDING)
    @patch("agents.memory_agent.get_uncompacted_interactions", new_callable=AsyncMock)
    async def test_handles_markdown_fenced_json(
        self, mock_get, mock_embed, mock_upsert, mock_mark
    ):
        """Should extract JSON from markdown code fences."""
        mock_get.return_value = [_make_interaction(i) for i in range(5)]

        fenced = f"```json\n{_make_nugget_response()}\n```"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=fenced)]
        mock_client.messages.create.return_value = mock_response

        from agents.memory_agent import MemoryAgent

        agent = MemoryAgent(anthropic_client=mock_client)
        count = await agent.compact_interactions()

        assert count == 1


class TestSearchMemories:
    """Tests for MemoryAgent.search_memories()."""

    @pytest.mark.asyncio
    @patch("agents.memory_agent.increment_memory_usage", new_callable=AsyncMock)
    @patch("agents.memory_agent.search_memory", new_callable=AsyncMock)
    @patch("agents.memory_agent.embed", return_value=FAKE_EMBEDDING)
    async def test_returns_top_2_and_increments_usage(
        self, mock_embed, mock_search, mock_increment
    ):
        """Should return top 2 memories and increment their usage counts."""
        rows = [_make_memory_row(0.92), _make_memory_row(0.88)]
        mock_search.return_value = rows

        from agents.memory_agent import MemoryAgent

        agent = MemoryAgent(anthropic_client=MagicMock())
        results = await agent.search_memories("iOS subscriptions")

        assert len(results) == 2
        assert mock_increment.call_count == 2
        mock_search.assert_called_once_with(FAKE_EMBEDDING, limit=2)


class TestTokenEfficiencyRouter:
    """Tests for MemoryAgent.get_context_for_query()."""

    @pytest.mark.asyncio
    @patch("agents.memory_agent.search_knowledge_base", new_callable=AsyncMock)
    @patch("agents.memory_agent.increment_memory_usage", new_callable=AsyncMock)
    @patch("agents.memory_agent.search_memory", new_callable=AsyncMock)
    @patch("agents.memory_agent.embed", return_value=FAKE_EMBEDDING)
    async def test_high_confidence_memory_only(
        self, mock_embed, mock_search_mem, mock_inc, mock_rag
    ):
        """Score > 0.85 should use memory only, no RAG call."""
        mock_search_mem.return_value = [_make_memory_row(0.92)]

        from agents.memory_agent import MemoryAgent

        agent = MemoryAgent(anthropic_client=MagicMock())
        context = await agent.get_context_for_query("iOS purchase")

        assert "iOS subscription purchase" in context
        mock_rag.assert_not_called()

    @pytest.mark.asyncio
    @patch("agents.memory_agent.search_knowledge_base", new_callable=AsyncMock)
    @patch("agents.memory_agent.increment_memory_usage", new_callable=AsyncMock)
    @patch("agents.memory_agent.search_memory", new_callable=AsyncMock)
    @patch("agents.memory_agent.embed", return_value=FAKE_EMBEDDING)
    async def test_medium_confidence_memory_plus_rag(
        self, mock_embed, mock_search_mem, mock_inc, mock_rag
    ):
        """Score 0.60-0.85 should use memory + 1 RAG chunk."""
        mock_search_mem.return_value = [_make_memory_row(0.75)]
        mock_rag.return_value = [
            {
                "content": "RAG doc content here",
                "source": "RevenueCat Docs",
                "url": "https://docs.revenuecat.com",
                "relevance_score": 0.80,
                "from_memory": False,
            }
        ]

        from agents.memory_agent import MemoryAgent

        agent = MemoryAgent(anthropic_client=MagicMock())
        context = await agent.get_context_for_query("iOS purchase")

        assert "iOS subscription purchase" in context
        assert "RAG doc content here" in context
        mock_rag.assert_called_once_with("iOS purchase", top_k=1)

    @pytest.mark.asyncio
    @patch("agents.memory_agent.search_knowledge_base", new_callable=AsyncMock)
    @patch("agents.memory_agent.increment_memory_usage", new_callable=AsyncMock)
    @patch("agents.memory_agent.search_memory", new_callable=AsyncMock)
    @patch("agents.memory_agent.embed", return_value=FAKE_EMBEDDING)
    async def test_low_confidence_full_rag(
        self, mock_embed, mock_search_mem, mock_inc, mock_rag
    ):
        """Score < 0.60 should use 3 RAG chunks only."""
        mock_search_mem.return_value = [_make_memory_row(0.40)]
        mock_rag.return_value = [
            {
                "content": f"RAG chunk {i}",
                "source": "RevenueCat Docs",
                "url": "https://docs.revenuecat.com",
                "relevance_score": 0.8 - i * 0.1,
                "from_memory": False,
            }
            for i in range(3)
        ]

        from agents.memory_agent import MemoryAgent

        agent = MemoryAgent(anthropic_client=MagicMock())
        context = await agent.get_context_for_query("something new")

        assert "RAG chunk 0" in context
        assert "RAG chunk 2" in context
        # Memory content should NOT appear in low-confidence path
        assert "iOS subscription purchase" not in context
        mock_rag.assert_called_once_with("something new", top_k=3)

    @pytest.mark.asyncio
    @patch("agents.memory_agent.search_knowledge_base", new_callable=AsyncMock)
    @patch("agents.memory_agent.search_memory", new_callable=AsyncMock)
    @patch("agents.memory_agent.embed", return_value=FAKE_EMBEDDING)
    async def test_no_memories_falls_to_rag(
        self, mock_embed, mock_search_mem, mock_rag
    ):
        """No memories at all should fall through to full RAG."""
        mock_search_mem.return_value = []
        mock_rag.return_value = [
            {
                "content": "doc content",
                "source": "Docs",
                "url": "",
                "relevance_score": 0.82,
                "from_memory": False,
            }
        ]

        from agents.memory_agent import MemoryAgent

        agent = MemoryAgent(anthropic_client=MagicMock())
        context = await agent.get_context_for_query("anything")

        assert "doc content" in context
        mock_rag.assert_called_once_with("anything", top_k=3)
