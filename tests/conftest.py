"""Shared test fixtures. Sets dummy env vars so config/settings.py loads."""

import os

# Set dummy env vars BEFORE any project module is imported.
# config/settings.py reads these at import time via os.environ.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("VOYAGE_API_KEY", "test-voyage-key")
os.environ.setdefault("E2B_API_KEY", "")
