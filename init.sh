#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Enable pgvector extension
    CREATE EXTENSION IF NOT EXISTS vector;

    -- Create full-access application user
    CREATE USER app_user WITH PASSWORD '${APP_USER_PASSWORD}';
    GRANT ALL PRIVILEGES ON DATABASE ${POSTGRES_DB} TO app_user;
    GRANT ALL PRIVILEGES ON SCHEMA public TO app_user;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public 
        GRANT ALL PRIVILEGES ON TABLES TO app_user;

    -- Create RAG user with limited access to embeddings table only
    CREATE USER rag_user WITH PASSWORD '${RAG_USER_PASSWORD}';
    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO rag_user;
    GRANT USAGE ON SCHEMA public TO rag_user;

    -- Create embeddings table
    CREATE TABLE IF NOT EXISTS embeddings (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        author TEXT,
        mimetype TEXT,
        category TEXT,
        source TEXT NOT NULL,
        url TEXT,
        parent_url TEXT,
        chunk_index INTEGER NOT NULL,
        total_chunks INTEGER NOT NULL,
        header TEXT,
        header_level INTEGER,
        markdown TEXT NOT NULL,
        hash TEXT NOT NULL,
        metadata JSONB,
        embedding vector(1536),
        created_at TIMESTAMP DEFAULT NOW(),
        last_modified TIMESTAMP DEFAULT NOW()
    );

    -- Grant full access to app_user
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app_user;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO app_user;

    -- Grant INSERT, SELECT, UPDATE, DELETE to RAG user
    GRANT SELECT, INSERT, UPDATE, DELETE ON embeddings TO rag_user;
    GRANT USAGE, SELECT ON SEQUENCE embeddings_id_seq TO rag_user;

    -- Create indexes
    CREATE INDEX idx_embeddings_source_url ON embeddings(source_url);

    -- HNSW index for better vector search performance
    CREATE INDEX idx_embeddings_hnsw ON embeddings 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
EOSQL
