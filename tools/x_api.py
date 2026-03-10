"""Twitter / X API integration via Tweepy."""

from __future__ import annotations

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


def search_recent_tweets(
    query: str, max_results: int = 20
) -> list[dict]:
    """Search recent tweets matching *query*.

    Returns a list of dicts with keys: id, text, author_id.
    """
    client = _get_client()
    resp = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=["author_id", "created_at"],
    )
    if not resp.data:
        return []
    return [
        {
            "id": str(tweet.id),
            "text": tweet.text,
            "author_id": str(tweet.author_id),
        }
        for tweet in resp.data
    ]


def reply_to_tweet(tweet_id: str, text: str) -> str:
    """Post a reply to *tweet_id*. Returns the new tweet id."""
    client = _get_client()
    resp = client.create_tweet(text=text, in_reply_to_tweet_id=tweet_id)
    return str(resp.data["id"])


def get_user_info(user_id: str) -> dict:
    """Return basic profile info for a Twitter user."""
    client = _get_client()
    resp = client.get_user(id=user_id, user_fields=["username", "name"])
    user = resp.data
    return {"id": str(user.id), "username": user.username, "name": user.name}
