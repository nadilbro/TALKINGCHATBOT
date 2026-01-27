# SQL/db_init.py
import os
import psycopg2

def _get_conn():
    """
    Uses DATABASE_URL if present, otherwise DB_* env vars.
    """
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=int(os.getenv("DB_PORT", "5432")),
        sslmode=os.getenv("DB_SSLMODE", "require"),  # Render often needs SSL
    )

def init_db() -> None:
    """
    Idempotent: safe to run multiple times.
    Creates pgvector + required tables + indexes if they don't exist.
    """
    conn = _get_conn()
    conn.autocommit = True

    with conn.cursor() as cur:
        # --- Extensions ---
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # --- Core tables ---
        # Your code uses: client_list(site_id, country, description, ...)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS client_list (
            site_id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT,
            description TEXT,
            country TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

        # Your code uses: allowed_websites(site_id, domain) + ON CONFLICT (site_id, domain)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS allowed_websites (
            site_id TEXT NOT NULL REFERENCES client_list(site_id) ON DELETE CASCADE,
            domain TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (site_id, domain)
        );
        """)

        # Embeddings table used by add_data/process_question
        # NOTE: set the vector dimension to match your embeddings model output.
        # Common dims:
        # - 1536 (older embedding models)
        # - 3072 (some newer)
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            id BIGSERIAL PRIMARY KEY,
            site_id TEXT NOT NULL REFERENCES client_list(site_id) ON DELETE CASCADE,
            source TEXT,
            chunk_index INT NOT NULL,
            data TEXT NOT NULL,
            embedding vector({embedding_dim}) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

        # Make chunk_index unique per site (so your MAX(chunk_index) logic is consistent)
        cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS embeddings_site_chunk_uniq
        ON embeddings (site_id, chunk_index);
        """)

        # Helpful for filtering by site
        cur.execute("""
        CREATE INDEX IF NOT EXISTS embeddings_site_idx
        ON embeddings (site_id);
        """)

        # Vector index (pgvector). Use IVFFLAT (needs ANALYZE and enough rows), or HNSW (if supported).
        # IVFFLAT:
        cur.execute("""
        CREATE INDEX IF NOT EXISTS embeddings_embedding_ivfflat
        ON embeddings USING ivfflat (embedding vector_l2_ops)
        WITH (lists = 100);
        """)

        # Optional: chatbot_settings table for your “save traits” endpoints
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chatbot_settings (
            site_id TEXT PRIMARY KEY REFERENCES client_list(site_id) ON DELETE CASCADE,
            chatbot_name TEXT,
            personality TEXT,
            tone TEXT,
            resp_length TEXT,
            temperature DOUBLE PRECISION,
            greeting TEXT,
            fallback TEXT,
            widget_color TEXT,
            widget_size TEXT,
            border_radius TEXT,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)
        #TEST INFORMATION FOR CHATBOT SETTINGS
        cur.execute('''
            INSERT INTO chatbot_settings (
                site_id,
                chatbot_name,
                personality,
                tone,
                resp_length,
                temperature,
                greeting,
                fallback,
                widget_color,
                widget_size,
                border_radius
            )
            VALUES (
                'site_abc123xyz789',
                'BubbleBot',
                'Friendly, helpful, slightly witty',
                'Casual',
                'medium',
                0.6,
                'Hey! How can I help you today?',
                'Sorry, I didn’t quite get that. Could you rephrase?',
                '#4F46E5',
                'medium',
                '16px'
            );
            
            ''')

    conn.close()
