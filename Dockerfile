# =============================================================================
# Multi-stage Dockerfile for RevenueCat AI Developer Advocate Agent
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# ---------------------------------------------------------------------------
# Stage 2: Runner — lean production image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runner

WORKDIR /app

# Runtime deps only (no gcc)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY agents/ agents/
COPY tools/ tools/
COPY dashboard/ dashboard/
COPY scripts/ scripts/
COPY database/ database/
COPY config/ config/

# Expose Streamlit port
EXPOSE 8501

# Entrypoint script picks mode from first argument:
#   docker run <image> dashboard   -> Streamlit
#   docker run <image> worker      -> Background agent loop
COPY <<'ENTRYPOINT' /app/entrypoint.sh
#!/bin/bash
set -e

MODE="${1:-worker}"

case "$MODE" in
  dashboard)
    echo "[entrypoint] Starting Streamlit dashboard..."
    exec streamlit run dashboard/app.py \
      --server.port "${PORT:-8501}" \
      --server.address 0.0.0.0 \
      --server.headless true
    ;;
  worker)
    echo "[entrypoint] Starting agent worker loop..."
    exec python -m agents.run_worker
    ;;
  *)
    echo "[entrypoint] Unknown mode: $MODE (use 'dashboard' or 'worker')"
    exit 1
    ;;
esac
ENTRYPOINT

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["worker"]
