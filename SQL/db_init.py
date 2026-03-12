# SQL/db_init.py
#TEMP FILE CAUSE AFTER IT IS RELEASED IT WONT NEED TO REDO THE INITIALISATION
import os
import psycopg2


def _get_conn():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=int(os.getenv("DB_PORT", "5432")),
        sslmode=os.getenv("DB_SSLMODE", "require"),
    )


def init_db() -> None:
    """
    Idempotent: safe to run multiple times.
    Creates ONLY:
      - accounts
      - sessions
      - messages
      - usage_events
    """
    conn = _get_conn()
    conn.autocommit = True

    with conn.cursor() as cur:
        # Useful for gen_random_uuid(); remove if your DB disallows extensions
        cur.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto";')

        # -----------------------------
        # 1) accounts  (AccountInit / ChatBotEdits)
        # -----------------------------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            user_id TEXT PRIMARY KEY,                  -- Firebase UID or your auth id
            name TEXT,
            email TEXT,
            phone TEXT,

            subscription_status TEXT,                  -- e.g. active/canceled/trialing
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,

            monthly_token_limit INT DEFAULT 0,
            monthly_token_used  INT DEFAULT 0,
            billing_cycle_start TIMESTAMPTZ,           -- when token cycle resets

            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS accounts_email_idx
        ON accounts (email);
        """)

        # -----------------------------
        # 2) sessions  (SessionInit / SessionCreate)
        # -----------------------------
        # Your SessionInit has chat_id (string), SessionCreate has id (string)
        # so we store session_id as TEXT (you generate it).
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,                       -- chat_id
            user_id TEXT NOT NULL REFERENCES accounts(user_id) ON DELETE CASCADE,

            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),

            rive_avatar TEXT,                          -- rive url or key
            avatar_voice TEXT,                          -- name of voiceOption
            last_message TEXT,
            welcome_message TEXT,

            status TEXT DEFAULT 'Open',                -- Open/Closed/etc
            summary TEXT,
            title TEXT
        );
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS sessions_user_time_idx
        ON sessions (user_id, created_at DESC);
        """)

        # -----------------------------
        # 3) messages  (for chat history)
        # -----------------------------
        # Even if you “haven’t done it”, you will need it to rebuild history properly.
        # Store role + content.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,

            role TEXT NOT NULL,                        -- user/assistant/system
            content TEXT NOT NULL,

            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS messages_session_time_idx
        ON messages (session_id, created_at ASC);
        """)

        # -----------------------------
        # 4) usage_events  (token usage / billing)
        # -----------------------------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS usage_events (
            usage_id BIGSERIAL PRIMARY KEY,

            user_id TEXT NOT NULL REFERENCES accounts(user_id) ON DELETE CASCADE,
            session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,

            provider TEXT,                             -- openai/gemini
            model TEXT,                                -- gpt-5-nano, gemini-2.0-flash, etc

            input_tokens INT DEFAULT 0,
            output_tokens INT DEFAULT 0,

            cost_cents INT DEFAULT 0,                  -- avoid floats

            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS usage_events_user_time_idx
        ON usage_events (user_id, created_at DESC);
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS usage_events_session_time_idx
        ON usage_events (session_id, created_at DESC);
        """)

        # -----------------------------
        # Optional seed data
        # -----------------------------
        seed = (os.getenv("DB_SEED_TEST_DATA", "1") or "1").strip() == "1"
        if seed:
            cur.execute("""
            INSERT INTO accounts (user_id, name, email, phone, subscription_status, monthly_token_limit, monthly_token_used)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO NOTHING;
            """, (
                "user_test_001", "Test User", "test@example.com", "0400000000",
                "active", 500000, 0
            ))

            cur.execute("""
            INSERT INTO sessions (id, user_id, title, welcome_message, status)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
            """, (
                "chat_test_001", "user_test_001", "Test Session",
                "Hey! How can I help you today?", "Open"
            ))

    conn.close()