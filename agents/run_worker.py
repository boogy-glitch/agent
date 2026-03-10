"""Background worker — runs the agent pipeline on a 5-minute loop.

Entry point: python -m agents.run_worker

Features:
- Graceful shutdown on SIGTERM / SIGINT
- Structured JSON logging
- X API failure: back off 15 minutes then retry
- Memory compaction runs alongside the main loop
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            entry["error"] = str(record.exc_info[1])
        return json.dumps(entry)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)

logger = logging.getLogger("worker")

# ---------------------------------------------------------------------------
# Import after logging is configured
# ---------------------------------------------------------------------------

from config.settings import settings, verify_env, print_banner  # noqa: E402
from agents.orchestrator import workflow  # noqa: E402
from agents.memory_agent import MemoryAgent  # noqa: E402

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown = asyncio.Event()


def _handle_signal(sig, frame):
    logger.info("Received %s — shutting down gracefully", signal.Signals(sig).name)
    _shutdown.set()


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

SCAN_INTERVAL = 5 * 60  # 5 minutes
BACKOFF_INTERVAL = 15 * 60  # 15 minutes on X API failure
COMPACTION_INTERVAL = settings.memory_compaction_interval_minutes * 60


async def run_scan(memory_agent: MemoryAgent) -> None:
    """Run one scan cycle: orchestrator pipeline + memory compaction."""
    try:
        logger.info("Starting scan cycle")
        result = await workflow.ainvoke({})
        status = result.get("status", "UNKNOWN")
        logger.info("Scan complete — status=%s", status)
    except Exception as exc:
        err_str = str(exc)
        if "tweepy" in err_str.lower() or "twitter" in err_str.lower() or "429" in err_str:
            logger.warning(
                "X API failure: %s — backing off %d minutes",
                err_str,
                BACKOFF_INTERVAL // 60,
            )
            await asyncio.sleep(BACKOFF_INTERVAL)
            return
        logger.exception("Scan cycle failed")


async def run_compaction(memory_agent: MemoryAgent) -> None:
    """Run one memory compaction cycle."""
    try:
        count = await memory_agent.compact_interactions()
        if count > 0:
            logger.info("Compacted %d memory nuggets", count)
    except Exception:
        logger.exception("Memory compaction failed")


async def main_loop() -> None:
    """Run the worker loop until shutdown signal."""
    verify_env()
    print_banner()

    memory_agent = MemoryAgent()
    last_compaction = 0.0

    logger.info(
        "Worker started — scan every %ds, compaction every %ds",
        SCAN_INTERVAL,
        COMPACTION_INTERVAL,
    )

    while not _shutdown.is_set():
        await run_scan(memory_agent)

        # Run compaction if interval has elapsed
        now = time.monotonic()
        if now - last_compaction >= COMPACTION_INTERVAL:
            await run_compaction(memory_agent)
            last_compaction = now

        # Wait for next scan or shutdown
        try:
            await asyncio.wait_for(_shutdown.wait(), timeout=SCAN_INTERVAL)
            break  # shutdown signalled
        except asyncio.TimeoutError:
            continue  # timeout = time for next scan

    logger.info("Worker stopped.")


def main() -> None:
    asyncio.run(main_loop())


if __name__ == "__main__":
    main()
