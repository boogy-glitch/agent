-- RevenueCat AI Developer Advocate Agent
-- PostgreSQL + pgvector schema

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Stores ingested RevenueCat documentation chunks with embeddings
CREATE TABLE knowledge_base (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    embedding vector(1536),
    source_url TEXT,
    section TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tracks every tweet interaction and its approval status
CREATE TABLE interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tweet_id TEXT UNIQUE NOT NULL,
    tweet_author TEXT NOT NULL,
    tweet_text TEXT NOT NULL,
    draft_reply TEXT,
    code_snippet TEXT,
    code_validated BOOLEAN DEFAULT FALSE,
    status TEXT DEFAULT 'PENDING_APPROVAL',
    compacted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Compacted memory nuggets from the Always-On Memory agent
CREATE TABLE memory_nuggets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    concept TEXT NOT NULL,
    summary TEXT NOT NULL,
    fix TEXT,
    embedding vector(1536),
    importance FLOAT DEFAULT 0.5,
    usage_count INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Weekly insight reports aggregating pain points
CREATE TABLE insight_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    week_start DATE NOT NULL,
    content TEXT NOT NULL,
    pain_points JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for vector similarity search
CREATE INDEX idx_knowledge_base_embedding ON knowledge_base
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_memory_nuggets_embedding ON memory_nuggets
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Indexes for common lookups
CREATE INDEX idx_interactions_status ON interactions (status);
CREATE INDEX idx_interactions_tweet_id ON interactions (tweet_id);
CREATE INDEX idx_insight_reports_week ON insight_reports (week_start);
CREATE INDEX idx_memory_nuggets_importance ON memory_nuggets (importance DESC);
CREATE INDEX idx_interactions_compacted ON interactions (compacted) WHERE compacted = FALSE;

-- RPC function to atomically increment memory nugget usage count
CREATE OR REPLACE FUNCTION increment_usage_count(row_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE memory_nuggets SET usage_count = usage_count + 1 WHERE id = row_id;
END;
$$ LANGUAGE plpgsql;
