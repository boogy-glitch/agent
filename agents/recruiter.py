"""Self-application system — the agent applies for its own job.

The "Agentic Handshake": find a real RevenueCat question on X, solve it
perfectly as proof of work, generate a portfolio page, and compose an
application tweet thread — all gated by human approval.

CLI:
    python -m agents.recruiter --apply    # Trigger application flow
    python -m agents.recruiter --preview  # Show what would be posted
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from config.settings import settings, VERSION
from database.db import get_latest_report, get_supabase, upsert_interaction

logger = logging.getLogger("recruiter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

_PORTFOLIO_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "portfolio.md"
_GITHUB_URL = "https://github.com/boogy-glitch/agent"


class RecruiterAgent:
    """Orchestrates the full self-application flow."""

    def __init__(self, anthropic_client: Anthropic | None = None):
        self._client = anthropic_client or Anthropic(
            api_key=settings.anthropic_api_key,
        )

    # ------------------------------------------------------------------
    # 1. Find application opportunity
    # ------------------------------------------------------------------

    async def find_application_opportunity(self) -> dict[str, Any]:
        """Search X for a recent unanswered RevenueCat question.

        Looks at tweets mentioning @RevenueCat or @jeiting that contain
        a technical question from a developer.
        """
        from tools.x_api import search_tweets

        # Search for questions near RevenueCat accounts
        queries = [
            ["@RevenueCat help", "RevenueCat SDK error"],
            ["@jeiting RevenueCat", "RevenueCat question"],
            ["StoreKit error RevenueCat", "IAP issue RevenueCat"],
        ]

        sb = get_supabase()
        existing_resp = sb.table("interactions").select("tweet_id").execute()
        existing_ids = (
            {r["tweet_id"] for r in existing_resp.data}
            if existing_resp.data
            else set()
        )

        for kws in queries:
            tweets = search_tweets(keywords=kws, max_results=10, hours=48)
            for tweet in tweets:
                if tweet["id"] not in existing_ids:
                    # Check it looks like a question
                    text = tweet["text"].lower()
                    if any(w in text for w in ("?", "help", "error", "issue", "how", "why")):
                        logger.info(
                            "Found opportunity: tweet %s from @%s",
                            tweet["id"],
                            tweet["author_username"],
                        )
                        return {
                            "tweet_id": tweet["id"],
                            "tweet_text": tweet["text"],
                            "tweet_author": tweet["author_username"],
                            "tweet_url": tweet["url"],
                        }

        logger.info("No unanswered opportunities found")
        return {}

    # ------------------------------------------------------------------
    # 2. Generate proof of work
    # ------------------------------------------------------------------

    async def generate_proof_of_work(self, tweet_data: dict) -> dict[str, Any]:
        """Run the full agent pipeline on the found tweet as a demonstration."""
        from agents.orchestrator import workflow

        logger.info("Generating proof of work for tweet %s", tweet_data.get("tweet_id"))

        result = await workflow.ainvoke(
            {
                "tweet_id": tweet_data["tweet_id"],
                "tweet_text": tweet_data["tweet_text"],
                "tweet_author": tweet_data["tweet_author"],
                "tweet_url": tweet_data.get("tweet_url", ""),
                "status": "SCOUTED",
                "_validation_attempts": 0,
            }
        )

        return {
            "tweet_id": tweet_data["tweet_id"],
            "tweet_text": tweet_data["tweet_text"],
            "tweet_author": tweet_data["tweet_author"],
            "draft_reply": result.get("edited_reply") or result.get("draft_reply", ""),
            "code_snippet": result.get("code_snippet", ""),
            "code_validated": result.get("code_validated", False),
            "status": result.get("status", "UNKNOWN"),
        }

    # ------------------------------------------------------------------
    # 3. Generate portfolio page
    # ------------------------------------------------------------------

    async def generate_portfolio_page(self) -> str:
        """Generate a Markdown portfolio page and save to disk."""
        stats = await self._gather_stats()
        nuggets = await self._get_recent_nuggets(5)
        samples = await self._get_sample_interactions(5)

        # Calculate validation success rate
        sb = get_supabase()
        validated_resp = (
            sb.table("interactions")
            .select("id", count="exact")
            .eq("code_validated", True)
            .execute()
        )
        total_with_code = (
            sb.table("interactions")
            .select("id", count="exact")
            .neq("code_snippet", "")
            .execute()
        )
        validation_rate = (
            (validated_resp.count / total_with_code.count * 100)
            if total_with_code.count
            else 100
        )

        # Build the portfolio Markdown
        lines = [
            f"# RevenueCat AI Developer Advocate — Portfolio",
            "",
            f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
            "",
            "## Agent Architecture",
            "",
            "```",
            "Twitter/X Scout (9 keywords, 24h window)",
            "       |",
            "       v",
            "LangGraph Orchestrator",
            "  |-- Architect Node (Claude Sonnet + prompt caching)",
            "  |     |-- Memory Agent (Token Efficiency Router)",
            "  |     |     |-- Score > 0.85: memory only (~200 tokens)",
            "  |     |     |-- Score 0.60-0.85: memory + 1 RAG chunk",
            "  |     |     |-- Score < 0.60: 3 RAG chunks (~2000 tokens)",
            "  |     |-- pgvector Knowledge Base (Voyage-3 embeddings)",
            "  |-- Validator Node (E2B sandbox + static analysis)",
            "  |-- Editor Node (Claude Haiku, authentic dev voice)",
            "  |-- HITL Dashboard (Streamlit, password-protected)",
            "       |",
            "       v",
            "Post to Twitter/X (human-approved only)",
            "```",
            "",
            "## Live Metrics",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Total interactions | {stats['total_interactions']} |",
            f"| Approved replies | {stats['approved_replies']} |",
            f"| Published to X | {stats['published_replies']} |",
            f"| Code validation rate | {validation_rate:.0f}% |",
            f"| Memory nuggets | {stats['memory_nuggets']} |",
            f"| Latest report | {stats['latest_report_date']} |",
            "",
            "## Last 5 Memory Nuggets",
            "",
            "*Demonstrating that the agent learns from past interactions:*",
            "",
        ]

        for i, n in enumerate(nuggets, 1):
            lines.append(
                f"{i}. **{n.get('concept', '?')}** "
                f"(importance: {n.get('importance', 0):.2f}, "
                f"used {n.get('usage_count', 0)}x)"
            )
            lines.append(f"   {n.get('summary', '')}")
            if n.get("fix"):
                lines.append(f"   Fix: `{n['fix'][:100]}`")
            lines.append("")

        lines.extend([
            "## Sample Interactions",
            "",
        ])

        for i, s in enumerate(samples, 1):
            validated_str = "Validated" if s.get("code_validated") else "Not validated"
            lines.append(f"### {i}. @{s.get('tweet_author', '?')}")
            lines.append(f"> {s.get('tweet_text', '')[:200]}")
            lines.append("")
            lines.append(f"**Reply:** {s.get('draft_reply', '')[:300]}")
            lines.append(f"**Status:** {s.get('status', '?')} | **Code:** {validated_str}")
            lines.append("")

        lines.extend([
            "## Links",
            "",
            f"- GitHub: [{_GITHUB_URL}]({_GITHUB_URL})",
            f"- Dashboard: *Available on request*",
            f"- Agent version: {VERSION}",
            "",
            "---",
            "*Built with LangGraph + pgvector + Claude Sonnet + E2B + Streamlit*",
        ])

        content = "\n".join(lines)
        _PORTFOLIO_PATH.write_text(content, encoding="utf-8")
        logger.info("Portfolio page saved to %s", _PORTFOLIO_PATH)
        return content

    # ------------------------------------------------------------------
    # 4. Compose application message
    # ------------------------------------------------------------------

    async def compose_application_message(
        self, proof: dict, portfolio_url: str
    ) -> list[str]:
        """Generate a 3-tweet application thread."""
        # Tweet 1: the actual technical answer (proof of work)
        tweet_1 = proof.get("draft_reply", "")
        # Truncate to 280 chars if needed
        if len(tweet_1) > 280:
            tweet_1 = tweet_1[:277] + "..."

        # Tweet 2: the reveal
        tweet_2 = (
            "Btw, I'm an experimental AI Developer Advocate "
            "built specifically for the @RevenueCat role. "
            "I just solved the above using persistent memory "
            f"+ E2B-validated code. Architecture: {portfolio_url}"
        )
        if len(tweet_2) > 280:
            tweet_2 = tweet_2[:277] + "..."

        # Tweet 3: the operator credit
        tweet_3 = (
            "Operator: @boogy_glitch | Stack: LangGraph + pgvector "
            "+ Claude Sonnet | Human approval: every post. "
            "DMs open for the RevenueCat team."
        )
        if len(tweet_3) > 280:
            tweet_3 = tweet_3[:277] + "..."

        return [tweet_1, tweet_2, tweet_3]

    # ------------------------------------------------------------------
    # 5. Submit application (NEVER auto-posts)
    # ------------------------------------------------------------------

    async def submit_application(self) -> dict[str, Any]:
        """Full application flow. Saves to DB for HUMAN APPROVAL only."""
        # Step 1: Find a tweet
        opportunity = await self.find_application_opportunity()
        if not opportunity:
            logger.warning("No application opportunity found. Aborting.")
            return {"status": "NO_OPPORTUNITY"}

        # Step 2: Generate proof of work
        proof = await self.generate_proof_of_work(opportunity)

        # Step 3: Generate portfolio
        portfolio_content = await self.generate_portfolio_page()
        portfolio_url = f"{_GITHUB_URL}/blob/main/dashboard/portfolio.md"

        # Step 4: Compose application tweets
        tweets = await self.compose_application_message(proof, portfolio_url)

        # Step 5: Save for human approval (NEVER auto-post)
        application_data = {
            "tweet_id": opportunity["tweet_id"],
            "tweet_author": opportunity["tweet_author"],
            "tweet_text": opportunity["tweet_text"],
            "draft_reply": json.dumps(
                {"thread": tweets, "proof": proof}, default=str
            ),
            "code_snippet": proof.get("code_snippet", ""),
            "code_validated": proof.get("code_validated", False),
            "status": "PENDING_APPLICATION",
        }

        interaction_id = await upsert_interaction(application_data)

        logger.info(
            "Application saved as PENDING_APPLICATION (id=%s). "
            "Check the dashboard to review and approve.",
            interaction_id,
        )

        return {
            "status": "PENDING_APPLICATION",
            "interaction_id": interaction_id,
            "opportunity": opportunity,
            "tweets": tweets,
            "portfolio_url": portfolio_url,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _gather_stats(self) -> dict:
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
        report = await get_latest_report()

        return {
            "total_interactions": total.count or 0,
            "approved_replies": approved.count or 0,
            "published_replies": published.count or 0,
            "memory_nuggets": nuggets.count or 0,
            "latest_report_date": report["week_start"] if report else "N/A",
        }

    async def _get_recent_nuggets(self, limit: int = 5) -> list[dict]:
        sb = get_supabase()
        resp = (
            sb.table("memory_nuggets")
            .select("concept, summary, fix, importance, usage_count")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []

    async def _get_sample_interactions(self, limit: int = 5) -> list[dict]:
        sb = get_supabase()
        resp = (
            sb.table("interactions")
            .select(
                "tweet_author, tweet_text, draft_reply, code_snippet, "
                "code_validated, status"
            )
            .in_("status", ["APPROVED", "PUBLISHED"])
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _run_apply() -> None:
    agent = RecruiterAgent()
    result = await agent.submit_application()
    print(json.dumps(result, indent=2, default=str))


async def _run_preview() -> None:
    agent = RecruiterAgent()

    print("=" * 60)
    print("APPLICATION PREVIEW (nothing will be posted)")
    print("=" * 60)

    # Step 1
    print("\n[1/4] Searching for application opportunity...")
    opportunity = await agent.find_application_opportunity()
    if not opportunity:
        print("No opportunity found. Using placeholder for preview.")
        opportunity = {
            "tweet_id": "preview_000",
            "tweet_text": "[Preview] How do I restore purchases with RevenueCat on iOS?",
            "tweet_author": "preview_user",
            "tweet_url": "https://x.com/preview_user/status/preview_000",
        }
    print(f"   Found: @{opportunity['tweet_author']}: {opportunity['tweet_text'][:80]}")

    # Step 2
    print("\n[2/4] Generating portfolio page...")
    portfolio = await agent.generate_portfolio_page()
    print(f"   Saved to {_PORTFOLIO_PATH} ({len(portfolio)} chars)")

    # Step 3
    print("\n[3/4] Composing application tweets...")
    portfolio_url = f"{_GITHUB_URL}/blob/main/dashboard/portfolio.md"
    tweets = await agent.compose_application_message(
        {
            "draft_reply": (
                "Call Purchases.shared.restorePurchases() — it syncs with "
                "Apple's servers and returns updated CustomerInfo with all "
                "active entitlements."
            ),
            "code_snippet": "Purchases.shared.restorePurchases { info, err in }",
            "code_validated": True,
        },
        portfolio_url,
    )

    for i, tweet in enumerate(tweets, 1):
        print(f"\n   Tweet {i} ({len(tweet)} chars):")
        print(f"   {tweet}")

    # Step 4
    print("\n[4/4] Stats:")
    stats = await agent._gather_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 60)
    print("Run with --apply to save for human approval.")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RevenueCat Agent — self-application system"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--apply", action="store_true", help="Trigger the full application flow"
    )
    group.add_argument(
        "--preview", action="store_true", help="Preview what would be posted"
    )
    args = parser.parse_args()

    if args.apply:
        asyncio.run(_run_apply())
    elif args.preview:
        asyncio.run(_run_preview())


if __name__ == "__main__":
    main()
