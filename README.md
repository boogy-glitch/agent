# RevenueCat AI Developer Advocate Agent

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/template)
[![CI / Deploy](https://github.com/boogy-glitch/agent/actions/workflows/deploy.yml/badge.svg)](https://github.com/boogy-glitch/agent/actions)

An autonomous AI agent that serves as a Developer Advocate for RevenueCat. It monitors Twitter/X for technical questions, generates grounded answers using RAG over official docs, validates code in a sandbox, and routes everything through a human-in-the-loop dashboard before posting.

## Architecture

```
                            +---------------------+
                            |   Twitter/X API     |
                            |  (Scout: 9 keywords)|
                            +----------+----------+
                                       |
                                       v
+----------------+         +-----------+-----------+         +------------------+
|  Firecrawl     |         |   LangGraph Orchestrator       |  Streamlit HITL  |
|  Doc Ingestion +-------->+                       +-------->+  Dashboard       |
|  (7 RC URLs)   |         |  scout -> architect   |         |                  |
+----------------+         |    -> validator        |         |  Approve / Edit  |
                           |    -> editor -> END   |         |  Reject / Regen  |
+----------------+         |                       |         +--------+---------+
|  pgvector      |<------->+  Memory Agent         |                  |
|  Knowledge Base|         |  (Token Efficiency    |                  v
|  + Memory      |         |   Router)             |         +-------+--------+
+----------------+         +-----------+-----------+         |  Post to X     |
                                       |                     +----------------+
                                       v
                           +-----------+-----------+
                           |  E2B Code Sandbox     |
                           |  + Static Analysis    |
                           |  (RC Method Registry) |
                           +-----------------------+

Weekly: generate_insights.py -> Claude Sonnet -> Slack + DB
Every 30m: Memory Agent -> compact interactions -> nuggets
```

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/boogy-glitch/agent.git
cd agent
cp .env.template .env
# Fill in API keys (see table below)
```

### 2. Start services

```bash
docker compose up -d
```

This launches PostgreSQL (pgvector + schema auto-applied), the worker (5-min scan loop), and the Streamlit dashboard on port 8501.

### 3. Ingest documentation

```bash
docker compose exec worker python scripts/ingest_docs.py --full
```

### 4. Open the dashboard

Navigate to `http://localhost:8501` to review and approve draft replies.

### 5. Generate a test report

```bash
docker compose exec worker python scripts/generate_insights.py --test
```

## Required API Keys

| Variable | Service | Required | Free Tier |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Claude (Sonnet + Haiku) | Yes | Pay-as-you-go |
| `SUPABASE_URL` | PostgreSQL + pgvector | Yes | 500 MB free |
| `SUPABASE_KEY` | Supabase auth | Yes | Free |
| `X_BEARER_TOKEN` | Twitter/X API v2 | For scanning | Free (Basic) |
| `X_API_KEY` | Twitter/X OAuth | For posting | Free (Basic) |
| `X_API_SECRET` | Twitter/X OAuth | For posting | Free (Basic) |
| `X_ACCESS_TOKEN` | Twitter/X OAuth | For posting | Free (Basic) |
| `X_ACCESS_TOKEN_SECRET` | Twitter/X OAuth | For posting | Free (Basic) |
| `E2B_API_KEY` | Code sandbox | Optional | 100 runs/mo free |
| `FIRECRAWL_API_KEY` | Doc crawling | For ingestion | 500 pages free |
| `VOYAGE_API_KEY` | Voyage-3 embeddings | Optional | Free tier |
| `SLACK_WEBHOOK_URL` | Weekly reports | Optional | Free |
| `DASHBOARD_PASSWORD` | Dashboard auth | Optional | N/A |

## Cost Breakdown

| Component | Model | Cost per interaction | Monthly (100 interactions) |
|---|---|---|---|
| Architect (reply gen) | Claude Sonnet | ~$0.010 | $1.00 |
| Editor (tone rewrite) | Claude Haiku | ~$0.003 | $0.30 |
| Memory compaction | Claude Haiku | ~$0.002 | $0.20 |
| Embeddings | Voyage-3 | ~$0.0001 | $0.01 |
| Code validation | E2B sandbox | Free (100/mo) | $0.00 |
| **Total** | | **~$0.015** | **~$1.51** |

Prompt caching reduces Sonnet input costs by ~90% for repeated system prompts.

## Deployment

### Railway (recommended)

1. Push to GitHub — Railway auto-detects `railway.toml`
2. Set environment variables in Railway dashboard
3. Two services deploy automatically: `dashboard` and `worker`

### Docker Compose (local / VPS)

```bash
docker compose up -d
```

Services: `dashboard` (port 8501), `worker` (background loop), `db` (PostgreSQL + pgvector).

### GitHub Actions

On every push to `main`:
1. Runs `pytest` test suite
2. Builds Docker image
3. Pushes to GitHub Container Registry (`ghcr.io`)
4. Deploys to Railway (requires `RAILWAY_TOKEN` secret)

## Project Structure

```
revenuecat-agent/
├── agents/
│   ├── orchestrator.py       # LangGraph pipeline: scout->architect->validator->editor
│   ├── memory_agent.py       # Always-On Memory with Token Efficiency Router
│   ├── run_worker.py         # Background worker with graceful shutdown
│   └── recruiter.py          # Self-application generator
├── tools/
│   ├── search_docs.py        # Voyage-3 embeddings + pgvector search
│   ├── validator.py          # E2B sandbox + static analysis fallback
│   └── x_api.py              # Twitter/X search, reply, mentions
├── dashboard/
│   ├── app.py                # HITL Control Center
│   └── pages/
│       ├── 1_Analytics.py    # Charts, costs, sentiment
│       └── 2_Memory_Bank.py  # Browse/edit/delete nuggets
├── scripts/
│   ├── ingest_docs.py        # Firecrawl + chunking + embedding
│   └── generate_insights.py  # Weekly reports + Slack delivery
├── database/
│   ├── schema.sql            # PostgreSQL + pgvector DDL
│   └── db.py                 # Async queries (SQLAlchemy + Supabase)
├── config/
│   └── settings.py           # Config + env validation + startup banner
├── tests/
│   ├── test_memory.py        # Memory compaction + router tests
│   └── test_validator.py     # Code validation tests
├── .github/workflows/
│   └── deploy.yml            # CI: test -> build -> push -> deploy
├── docker-compose.yml        # Local 3-service stack
├── Dockerfile                # Multi-stage build
├── railway.toml              # Railway deployment config
└── pyproject.toml            # Dependencies (Python 3.12+)
```

## Development

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
python -m pytest tests/ -v
```
