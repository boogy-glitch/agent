"""Product Intelligence — weekly report generator.

Analyses the past 7 days of interactions, groups by platform / error type /
topic, generates an executive report via Claude Sonnet, posts to Slack,
and saves to the database.

CLI:
    python scripts/generate_insights.py --now    # Run immediately
    python scripts/generate_insights.py --test   # Use mock data
    python scripts/generate_insights.py          # Start Sunday 09:00 UTC scheduler
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import schedule

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anthropic import Anthropic

from config.settings import settings
from database.db import get_supabase, insert_insight_report

logger = logging.getLogger("insights")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

_client = Anthropic(api_key=settings.anthropic_api_key)

# ---------------------------------------------------------------------------
# Cached system prompt
# ---------------------------------------------------------------------------

_REPORT_SYSTEM = [
    {
        "type": "text",
        "text": (
            "You are a Senior Product Manager at RevenueCat.\n"
            "Write a concise weekly developer community report.\n\n"
            "Format (use Markdown):\n"
            "# RevenueCat Community Report — Week of [DATE]\n\n"
            "## Executive Summary\n"
            "[2-3 sentences about the week]\n\n"
            "## Top 3 Developer Pain Points\n"
            "1. **[Pain Point]**: X developers affected — [Recommendation]\n"
            "2. ...\n"
            "3. ...\n\n"
            "## Trend Alert\n"
            "[Something unusual detected this week]\n\n"
            "## Recommended Product Actions\n"
            "- [ ] Update documentation for [section]\n"
            "- [ ] Fix or clarify [feature]\n\n"
            "Tone: Professional but conversational. Max 600 words.\n"
            "Return ONLY the Markdown report, no extra text."
        ),
        "cache_control": {"type": "ephemeral"},
    }
]


# ---------------------------------------------------------------------------
# Platform / topic detection helpers
# ---------------------------------------------------------------------------

_PLATFORM_KEYWORDS: dict[str, list[str]] = {
    "iOS": ["swift", "storekit", "ios", "xcode", "skerror", "apple"],
    "Android": ["kotlin", "android", "billing", "play store", "google play"],
    "Flutter": ["flutter", "dart"],
    "React Native": ["react native", "react-native"],
    "Unity": ["unity", "c#"],
}

_SENTIMENT_NEGATIVE = [
    "error", "bug", "broken", "fail", "crash", "issue", "problem",
    "frustrated", "stuck", "help", "fix", "wrong", "doesn't work",
]
_SENTIMENT_POSITIVE = [
    "thanks", "solved", "works", "great", "awesome", "perfect", "love",
]


def _detect_platform(text: str) -> str:
    t = text.lower()
    for platform, kws in _PLATFORM_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return platform
    return "General"


def _detect_sentiment(text: str) -> str:
    t = text.lower()
    neg = sum(1 for w in _SENTIMENT_NEGATIVE if w in t)
    pos = sum(1 for w in _SENTIMENT_POSITIVE if w in t)
    if neg > pos:
        return "frustrated" if neg >= 3 else "negative"
    if pos > neg:
        return "positive"
    return "neutral"


def _extract_topic(text: str) -> str:
    """Rough topic extraction from tweet text."""
    t = text.lower()
    topics = {
        "StoreKit 2 migration": ["storekit 2", "sk2", "storekit2"],
        "Subscription status": ["subscription status", "entitlement", "customerinfo"],
        "Purchase flow": ["purchase", "buy", "checkout", "payment"],
        "Restore purchases": ["restore", "restoring"],
        "Configuration": ["configure", "setup", "api key", "sdk init"],
        "Offerings": ["offering", "package", "paywall"],
        "Billing issues": ["billing", "charge", "refund", "receipt"],
        "Web SDK": ["web", "stripe", "webhook"],
    }
    for topic, kws in topics.items():
        if any(kw in t for kw in kws):
            return topic
    return "Other"


# ---------------------------------------------------------------------------
# 1. Analyze weekly interactions
# ---------------------------------------------------------------------------


def analyze_weekly_interactions(interactions: list[dict] | None = None) -> dict:
    """Query last 7 days and return structured analytics data."""
    if interactions is None:
        sb = get_supabase()
        since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        resp = (
            sb.table("interactions")
            .select("tweet_text, draft_reply, code_snippet, status, tweet_author, created_at")
            .gte("created_at", since)
            .order("created_at", desc=False)
            .execute()
        )
        interactions = resp.data or []

    if not interactions:
        return {
            "total": 0,
            "week_start": (date.today() - timedelta(days=date.today().weekday())).isoformat(),
            "platforms": {},
            "topics": {},
            "sentiments": {},
            "statuses": {},
            "pain_points": [],
            "interactions_sample": [],
        }

    platforms: Counter = Counter()
    topics: Counter = Counter()
    sentiments: Counter = Counter()
    statuses: Counter = Counter()

    for ix in interactions:
        text = ix.get("tweet_text", "")
        platforms[_detect_platform(text)] += 1
        topics[_extract_topic(text)] += 1
        sentiments[_detect_sentiment(text)] += 1
        statuses[ix.get("status", "UNKNOWN")] += 1

    # Build pain-points list from topic frequency
    pain_points = [
        {"topic": topic, "count": count, "description": ""}
        for topic, count in topics.most_common(5)
        if topic != "Other"
    ]

    # Sample interactions for LLM context (limit to 20 to save tokens)
    sample = [
        {
            "tweet": ix.get("tweet_text", ""),
            "reply": ix.get("draft_reply", ""),
            "status": ix.get("status", ""),
        }
        for ix in interactions[:20]
    ]

    week_start = date.today() - timedelta(days=date.today().weekday())

    return {
        "total": len(interactions),
        "week_start": week_start.isoformat(),
        "platforms": dict(platforms.most_common()),
        "topics": dict(topics.most_common()),
        "sentiments": dict(sentiments.most_common()),
        "statuses": dict(statuses.most_common()),
        "pain_points": pain_points,
        "interactions_sample": sample,
    }


# ---------------------------------------------------------------------------
# 2. Generate insight report via Claude
# ---------------------------------------------------------------------------


async def generate_insight_report(data: dict) -> str:
    """Call Claude Sonnet to produce a Markdown weekly report."""
    # Remove large sample from the JSON sent to Claude to save tokens
    data_for_llm = {k: v for k, v in data.items() if k != "interactions_sample"}
    data_for_llm["sample_interactions"] = data.get("interactions_sample", [])[:10]

    response = _client.messages.create(
        model=settings.claude_model,
        max_tokens=2048,
        system=_REPORT_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Generate the weekly community report from this data:\n\n"
                    f"```json\n{json.dumps(data_for_llm, indent=2)}\n```"
                ),
            }
        ],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# 3. Send to Slack
# ---------------------------------------------------------------------------


def send_report_to_slack(report: str) -> bool:
    """POST the report to the configured Slack webhook.

    Returns True on success, False otherwise.
    """
    url = settings.slack_webhook_url
    if not url:
        logger.info("No SLACK_WEBHOOK_URL configured — skipping Slack delivery.")
        return False

    # Convert Markdown headings to Slack block format
    sections: list[str] = re.split(r"\n(?=##? )", report)

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "RevenueCat Community Report",
            },
        },
    ]

    for section in sections:
        text = section.strip()
        if not text:
            continue
        # Slack mrkdwn has a 3000 char limit per block
        if len(text) > 2900:
            text = text[:2900] + "..."
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": text},
            }
        )

    blocks.append({"type": "divider"})
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        "Generated by *RevenueCat AI Developer Advocate* "
                        f"| {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
                    ),
                }
            ],
        }
    )

    try:
        resp = httpx.post(url, json={"blocks": blocks}, timeout=15)
        resp.raise_for_status()
        logger.info("Report posted to Slack.")
        return True
    except Exception as exc:
        logger.error("Failed to post to Slack: %s", exc)
        return False


# ---------------------------------------------------------------------------
# 4. Save to database
# ---------------------------------------------------------------------------


async def save_report_to_db(report: str, data: dict) -> str:
    """Persist the report and pain points in the insight_reports table."""
    week_start = date.fromisoformat(data["week_start"])
    pain_points = data.get("pain_points", [])
    report_id = await insert_insight_report(week_start, report, pain_points)
    logger.info("Report %s saved for week of %s", report_id, week_start)
    return report_id


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(interactions: list[dict] | None = None) -> str:
    """Execute the full insights pipeline: analyse -> generate -> slack -> DB."""
    logger.info("Starting insights pipeline...")

    data = analyze_weekly_interactions(interactions)
    if data["total"] == 0:
        logger.info("No interactions in the past week. Skipping.")
        return ""

    logger.info(
        "Analysed %d interactions: %s",
        data["total"],
        json.dumps(data["platforms"]),
    )

    report = await generate_insight_report(data)
    logger.info("Report generated (%d chars)", len(report))

    print("\n" + report + "\n")

    send_report_to_slack(report)
    report_id = await save_report_to_db(report, data)

    logger.info("Pipeline complete. Report ID: %s", report_id)
    return report_id


# ---------------------------------------------------------------------------
# Mock data for --test mode
# ---------------------------------------------------------------------------

_MOCK_INTERACTIONS: list[dict] = [
    {
        "tweet_text": "Getting SKError 0 when trying to purchase a subscription on iOS. Purchases.shared.purchase keeps failing. Anyone seen this?",
        "draft_reply": "SKError 0 usually means the StoreKit environment isn't configured. Make sure you call Purchases.configure(withAPIKey:) in didFinishLaunching.",
        "code_snippet": "Purchases.configure(withAPIKey: \"appl_xxxxx\")",
        "status": "APPROVED",
        "tweet_author": "ios_dev_42",
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "tweet_text": "RevenueCat Flutter SDK: Purchases.getOfferings() returns empty. What am I missing?",
        "draft_reply": "Empty offerings usually means products aren't configured in the RC dashboard. Check your Offerings and make sure at least one Package has a product attached.",
        "code_snippet": "final offerings = await Purchases.getOfferings();",
        "status": "APPROVED",
        "tweet_author": "flutter_fanatic",
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "tweet_text": "How do I restore purchases with RevenueCat on Android? Using Kotlin and BillingClient.",
        "draft_reply": "Call Purchases.sharedInstance.restorePurchases(). It syncs with Google Play and returns the updated CustomerInfo.",
        "code_snippet": "Purchases.sharedInstance.restorePurchases { customerInfo -> }",
        "status": "APPROVED",
        "tweet_author": "kotlin_krafter",
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "tweet_text": "StoreKit 2 migration with RevenueCat — do I need to change anything? Still using the old StoreKit APIs.",
        "draft_reply": "RevenueCat SDK v4+ handles StoreKit 2 under the hood. Just update to the latest SDK version and it will automatically use SK2 on iOS 15+.",
        "code_snippet": "",
        "status": "APPROVED",
        "tweet_author": "swifty_dev",
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "tweet_text": "RevenueCat help — subscription billing not working after migrating from direct StoreKit. Customers losing access.",
        "draft_reply": "After migration, call Purchases.shared.logIn() with your user IDs to link existing subscribers. Then call restorePurchases() to sync entitlements.",
        "code_snippet": 'Purchases.shared.logIn("user_id") { customerInfo, created, error in }',
        "status": "APPROVED",
        "tweet_author": "indie_ios",
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "tweet_text": "Is there a way to get customer info without a network call in RevenueCat? Need it for offline mode.",
        "draft_reply": "CustomerInfo is cached locally by the SDK. Just call getCustomerInfo() — it returns the cached version instantly and refreshes in the background.",
        "code_snippet": "",
        "status": "PENDING_APPROVAL",
        "tweet_author": "offline_first",
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "tweet_text": "RevenueCat React Native: TypeError when calling configure. Frustrating.",
        "draft_reply": "Make sure you're calling configure() after the native modules are loaded. In React Native, use it inside useEffect or after the app is initialized.",
        "code_snippet": "",
        "status": "APPROVED",
        "tweet_author": "rn_builder",
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def _scheduled_run() -> None:
    """Synchronous entry point for the scheduler."""
    asyncio.run(run_pipeline())


def start_scheduler() -> None:
    """Run the report generator every Sunday at 09:00 UTC."""
    schedule.every().sunday.at("09:00").do(_scheduled_run)
    logger.info("Insight report scheduler started (every Sunday 09:00 UTC)")
    while True:
        schedule.run_pending()
        time.sleep(60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RevenueCat weekly insights report generator"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--now", action="store_true", help="Generate report immediately"
    )
    group.add_argument(
        "--test", action="store_true", help="Generate report using mock data"
    )

    args = parser.parse_args()

    if args.now:
        asyncio.run(run_pipeline())
    elif args.test:
        logger.info("Running with mock data...")
        asyncio.run(run_pipeline(interactions=_MOCK_INTERACTIONS))
    else:
        start_scheduler()


if __name__ == "__main__":
    main()
