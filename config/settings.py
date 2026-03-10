"""Configuration and environment variable loading."""

import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Required env vars (checked at startup)
# ---------------------------------------------------------------------------

_REQUIRED_VARS = [
    "ANTHROPIC_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY",
]

_OPTIONAL_VARS = [
    "X_API_KEY",
    "X_API_SECRET",
    "X_ACCESS_TOKEN",
    "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "E2B_API_KEY",
    "FIRECRAWL_API_KEY",
    "VOYAGE_API_KEY",
    "SLACK_WEBHOOK_URL",
    "DASHBOARD_PASSWORD",
]


@dataclass(frozen=True)
class Settings:
    # Anthropic / Claude
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    claude_model: str = "claude-sonnet-4-20250514"
    claude_model_haiku: str = "claude-haiku-4-5-20251001"

    # Supabase (Postgres + pgvector)
    supabase_url: str = field(
        default_factory=lambda: os.environ.get("SUPABASE_URL", "")
    )
    supabase_key: str = field(
        default_factory=lambda: os.environ.get("SUPABASE_KEY", "")
    )

    # Twitter / X
    x_api_key: str = field(
        default_factory=lambda: os.environ.get("X_API_KEY", "")
    )
    x_api_secret: str = field(
        default_factory=lambda: os.environ.get("X_API_SECRET", "")
    )
    x_access_token: str = field(
        default_factory=lambda: os.environ.get("X_ACCESS_TOKEN", "")
    )
    x_access_token_secret: str = field(
        default_factory=lambda: os.environ.get("X_ACCESS_TOKEN_SECRET", "")
    )
    x_bearer_token: str = field(
        default_factory=lambda: os.environ.get("X_BEARER_TOKEN", "")
    )

    # E2B code sandbox
    e2b_api_key: str = field(
        default_factory=lambda: os.environ.get("E2B_API_KEY", "")
    )

    # Firecrawl (docs ingestion)
    firecrawl_api_key: str = field(
        default_factory=lambda: os.environ.get("FIRECRAWL_API_KEY", "")
    )

    # Voyage AI (embeddings)
    voyage_api_key: str = field(
        default_factory=lambda: os.environ.get("VOYAGE_API_KEY", "")
    )

    # Embedding config
    embedding_model: str = "voyage-3"
    embedding_model_fallback: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Ingestion
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 100

    # Agent behaviour
    max_reply_length: int = 280
    similarity_threshold: float = 0.78
    memory_compaction_interval_hours: int = 6
    memory_compaction_interval_minutes: int = 30
    memory_compaction_min_interactions: int = 5
    memory_high_confidence: float = 0.85
    memory_low_confidence: float = 0.60

    # Slack
    slack_webhook_url: str = field(
        default_factory=lambda: os.environ.get("SLACK_WEBHOOK_URL", "")
    )

    # Dashboard
    dashboard_password: str = field(
        default_factory=lambda: os.environ.get("DASHBOARD_PASSWORD", "")
    )

    # RevenueCat docs URLs to ingest
    docs_urls: tuple[str, ...] = (
        "https://www.revenuecat.com/docs/",
        "https://www.revenuecat.com/docs/getting-started",
        "https://www.revenuecat.com/docs/ios",
        "https://www.revenuecat.com/docs/android",
        "https://www.revenuecat.com/docs/flutter",
        "https://www.revenuecat.com/docs/react-native",
        "https://www.revenuecat.com/docs/unity",
    )


settings = Settings()


# ---------------------------------------------------------------------------
# Startup health check
# ---------------------------------------------------------------------------


def verify_env() -> None:
    """Verify all required environment variables are set.

    Prints a clear error message and exits with code 1 if any are missing.
    """
    missing = [v for v in _REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        print("=" * 60, file=sys.stderr)
        print("FATAL: Missing required environment variables:", file=sys.stderr)
        for v in missing:
            print(f"  - {v}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Copy .env.template to .env and fill in the values.", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)

    # Warn about optional vars
    unset = [v for v in _OPTIONAL_VARS if not os.environ.get(v)]
    if unset:
        print(f"Warning: {len(unset)} optional env var(s) not set: {', '.join(unset)}")


def print_banner() -> None:
    """Print startup banner with version and config summary."""
    configured = []
    if settings.x_bearer_token:
        configured.append("Twitter/X")
    if settings.e2b_api_key:
        configured.append("E2B sandbox")
    if settings.firecrawl_api_key:
        configured.append("Firecrawl")
    if settings.voyage_api_key:
        configured.append("Voyage embeddings")
    if settings.slack_webhook_url:
        configured.append("Slack")

    print("=" * 60)
    print(f"  RevenueCat AI Developer Advocate Agent v{VERSION}")
    print("=" * 60)
    print(f"  Model (main):     {settings.claude_model}")
    print(f"  Model (fast):     {settings.claude_model_haiku}")
    print(f"  Embedding:        {settings.embedding_model}")
    print(f"  Supabase:         {settings.supabase_url[:40]}...")
    print(f"  Integrations:     {', '.join(configured) or 'none'}")
    print(f"  Scan interval:    5 minutes")
    print(f"  Compaction:       every {settings.memory_compaction_interval_minutes} min")
    print("=" * 60)
