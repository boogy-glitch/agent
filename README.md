# RevenueCat AI Developer Advocate Agent

An autonomous AI agent that serves as a Developer Advocate for RevenueCat. It monitors Twitter/X for technical questions about RevenueCat SDKs, generates grounded answers using RAG over the official docs, validates code snippets in a sandbox, and routes everything through a human-in-the-loop approval dashboard before posting.

## Architecture

```
Twitter/X  ──>  LangGraph Orchestrator  ──>  Streamlit HITL Dashboard
                      │                              │
                      ├── Classify tweet              ├── Approve / Reject / Edit
                      ├── RAG: pgvector search        └── Post reply to Twitter
                      ├── Draft reply (Claude)
                      ├── Validate code (E2B)
                      └── Save interaction
```

### Core Components

**Multi-Agent Orchestrator** (`agents/orchestrator.py`)
A LangGraph state machine that coordinates the full pipeline: tweet classification, knowledge retrieval, reply drafting, code validation, and posting. Conditional edges skip irrelevant tweets and bypass validation when no code is present.

**Knowledge Base** (`tools/search_docs.py` + `database/`)
RevenueCat documentation is crawled via Firecrawl, chunked, embedded (OpenAI `text-embedding-3-small`, 1536 dims), and stored in PostgreSQL with pgvector. Semantic search retrieves the most relevant chunks at query time.

**Code Validator** (`tools/validator.py`)
Every code snippet in a draft reply is executed inside an E2B cloud sandbox before it reaches a human reviewer. The result (pass/fail + output) is stored alongside the interaction.

**Always-On Memory Agent** (`agents/memory_agent.py`)
A background loop that periodically scans recent interactions, uses Claude to extract recurring patterns and pain-points, and compacts them into *memory nuggets* with embeddings. These nuggets are retrieved during future interactions to improve response quality over time.

**HITL Dashboard** (`dashboard/app.py`)
A Streamlit app where a human reviewer can approve, reject, or edit draft replies before they are posted. Includes a product-insights page (`dashboard/pages/insights.py`) showing weekly reports, pain-point trends, and memory-nugget statistics.

**Recruiter / Self-Application** (`agents/recruiter.py`)
Generates a polished application document showcasing the agent's real metrics and architecture, intended for the RevenueCat Developer Advocate position.

**Weekly Insight Reports** (`scripts/generate_insights.py`)
Aggregates the past week's interactions, extracts pain-point themes via Claude, and stores a structured report with JSONB pain-point data for the dashboard.

### Database Schema (PostgreSQL + pgvector)

| Table | Purpose |
|---|---|
| `knowledge_base` | Embedded documentation chunks for RAG |
| `interactions` | Every tweet interaction with draft, status, validation result |
| `memory_nuggets` | Compacted reusable knowledge with embeddings |
| `insight_reports` | Weekly summaries with JSONB pain-point arrays |

### Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude (via `langchain-anthropic`) |
| Orchestration | LangGraph |
| Database | PostgreSQL + pgvector (Supabase) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Code sandbox | E2B |
| Docs ingestion | Firecrawl |
| Social API | Tweepy (Twitter/X) |
| Dashboard | Streamlit |
| Deployment | Docker Compose |

## Quick Start

1. **Clone and configure**
   ```bash
   cp .env.template .env
   # Fill in all API keys in .env
   ```

2. **Start services**
   ```bash
   docker compose up -d
   ```
   This launches PostgreSQL (with pgvector and the schema auto-applied), the agent process, and the Streamlit dashboard on port 8501.

3. **Ingest documentation**
   ```bash
   docker compose exec agent python scripts/ingest_docs.py
   ```

4. **Open the dashboard**
   Navigate to `http://localhost:8501` to review and approve draft replies.

5. **Generate a weekly report**
   ```bash
   docker compose exec agent python scripts/generate_insights.py
   ```

## Development

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```

## Project Structure

```
revenuecat-agent/
├── agents/               # LangGraph agents and workflows
│   ├── orchestrator.py   # Main multi-agent pipeline
│   ├── memory_agent.py   # Background memory compaction
│   └── recruiter.py      # Self-application generator
├── tools/                # Standalone tool modules
│   ├── search_docs.py    # pgvector semantic search
│   ├── validator.py      # E2B code sandbox
│   └── x_api.py          # Twitter/X API wrapper
├── dashboard/            # Streamlit HITL interface
│   ├── app.py            # Main approval dashboard
│   └── pages/insights.py # Product insights page
├── scripts/              # One-off and scheduled scripts
│   ├── ingest_docs.py    # Crawl & embed RevenueCat docs
│   └── generate_insights.py
├── database/             # Schema and DB helpers
│   ├── schema.sql        # PostgreSQL + pgvector DDL
│   └── db.py             # Async queries and connections
├── config/settings.py    # Centralised configuration
├── docker-compose.yml    # Full-stack local deployment
├── Dockerfile
└── pyproject.toml
```
