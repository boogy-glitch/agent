"""Twitter / X API integration via Tweepy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import tweepy

from config.settings import settings

_client: tweepy.Client | None = None


def _get_client() -> tweepy.Client:
    global _client
    if _client is None:
        _client = tweepy.Client(
            bearer_token=settings.x_bearer_token,
            consumer_key=settings.x_api_key,
            consumer_secret=settings.x_api_secret,
            access_token=settings.x_access_token,
            access_token_secret=settings.x_access_token_secret,
            wait_on_rate_limit=True,
        )
    return _client


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

# Keywords the scout node monitors
SCOUT_KEYWORDS: list[str] = [
    "StoreKit error",
    "RevenueCat help",
    "IAP issue",
    "in-app purchase",
    "subscription billing",
    "SKError",
    "RevenueCat SDK",
    "Purchases.configure",
    "billing client",
]


def search_tweets(
    keywords: list[str] | None = None,
    max_results: int = 20,
    hours: int = 24,
) -> list[dict]:
    """Search recent tweets matching any of *keywords* (OR query).

    Filters to the last *hours* hours and excludes retweets.
    Returns a list of dicts with keys:
        id, text, author_id, author_username, created_at, url
    """
    keywords = keywords or SCOUT_KEYWORDS
    # Build an OR query: ("term1" OR "term2" ...) -is:retweet
    or_clause = " OR ".join(f'"{kw}"' for kw in keywords)
    query = f"({or_clause}) -is:retweet lang:en"

    client = _get_client()
    start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

    resp = client.search_recent_tweets(
        query=query,
        max_results=min(max_results, 100),
        start_time=start_time,
        tweet_fields=["author_id", "created_at", "conversation_id"],
        user_fields=["username", "public_metrics"],
        expansions=["author_id"],
    )

    if not resp.data:
        return []

    # Build a user lookup map from includes
    users: dict[str, dict] = {}
    if resp.includes and "users" in resp.includes:
        for u in resp.includes["users"]:
            users[str(u.id)] = {
                "username": u.username,
                "followers": u.public_metrics.get("followers_count", 0)
                if u.public_metrics
                else 0,
            }

    results: list[dict] = []
    for tweet in resp.data:
        author_id = str(tweet.author_id)
        user_info = users.get(author_id, {})
        followers = user_info.get("followers", 0)

        # Filter: minimum 10 followers
        if followers < 10:
            continue

        username = user_info.get("username", "unknown")
        results.append(
            {
                "id": str(tweet.id),
                "text": tweet.text,
                "author_id": author_id,
                "author_username": username,
                "created_at": tweet.created_at.isoformat() if tweet.created_at else "",
                "url": f"https://x.com/{username}/status/{tweet.id}",
            }
        )

    return results


# ---------------------------------------------------------------------------
# Post / Reply
# ---------------------------------------------------------------------------


def post_reply(tweet_id: str, text: str) -> dict:
    """Post a reply to *tweet_id*.

    Returns a dict with:
        id  - the new tweet's id
        url - direct link to the reply
    """
    client = _get_client()
    resp = client.create_tweet(text=text, in_reply_to_tweet_id=tweet_id)
    new_id = str(resp.data["id"])
    return {
        "id": new_id,
        "url": f"https://x.com/i/status/{new_id}",
    }


# Legacy alias used by other modules
def reply_to_tweet(tweet_id: str, text: str) -> str:
    """Post a reply and return just the new tweet id (legacy API)."""
    return post_reply(tweet_id, text)["id"]


# ---------------------------------------------------------------------------
# Mentions
# ---------------------------------------------------------------------------


def get_user_mentions(max_results: int = 20) -> list[dict]:
    """Fetch recent mentions of the authenticated user.

    Returns a list of dicts with keys: id, text, author_id, created_at.
    """
    client = _get_client()
    # Get authenticated user id
    me = client.get_me()
    if not me or not me.data:
        return []

    resp = client.get_users_mentions(
        id=me.data.id,
        max_results=min(max_results, 100),
        tweet_fields=["author_id", "created_at", "conversation_id"],
    )
    if not resp.data:
        return []
    return [
        {
            "id": str(tweet.id),
            "text": tweet.text,
            "author_id": str(tweet.author_id),
            "created_at": tweet.created_at.isoformat() if tweet.created_at else "",
        }
        for tweet in resp.data
    ]


# ---------------------------------------------------------------------------
# User info
# ---------------------------------------------------------------------------


def get_user_info(user_id: str) -> dict:
    """Return basic profile info for a Twitter user."""
    client = _get_client()
    resp = client.get_user(
        id=user_id,
        user_fields=["username", "name", "public_metrics"],
    )
    user = resp.data
    return {
        "id": str(user.id),
        "username": user.username,
        "name": user.name,
        "followers": user.public_metrics.get("followers_count", 0)
        if user.public_metrics
        else 0,
    }
