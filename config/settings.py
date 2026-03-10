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

    # Embedding config
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Agent behaviour
    max_reply_length: int = 280
    similarity_threshold: float = 0.78
    memory_compaction_interval_hours: int = 6


settings = Settings()
