"""Configuration and environment variable loading."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Anthropic / Claude
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ["ANTHROPIC_API_KEY"]
    )
    claude_model: str = "claude-sonnet-4-20250514"
    claude_model_haiku: str = "claude-haiku-4-5-20251001"

    # Supabase (Postgres + pgvector)
    supabase_url: str = field(
        default_factory=lambda: os.environ["SUPABASE_URL"]
    )
    supabase_key: str = field(
        default_factory=lambda: os.environ["SUPABASE_KEY"]
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
