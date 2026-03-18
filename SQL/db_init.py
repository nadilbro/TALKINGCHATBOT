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
      - rive_avatars
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
        # 2) rive_avatars  (must be created before sessions, which FK into it)
        # -----------------------------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS rive_avatars (
            avatar_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,

            name TEXT NOT NULL UNIQUE,                 -- display name of the character
            voice TEXT NOT NULL,                       -- avatar voice (choose from microsoft AZURE)
            url TEXT NOT NULL,                         -- URL to the .riv file
            prompt TEXT,                               -- system prompt / persona for this character
            version TEXT NOT NULL DEFAULT '1.0',       -- asset/schema version

            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS rive_avatars_name_idx
        ON rive_avatars (name);
        """)

        # -----------------------------
        # 3) sessions  (SessionInit / SessionCreate)
        # -----------------------------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,                       -- chat_id
            user_id TEXT NOT NULL REFERENCES accounts(user_id) ON DELETE CASCADE,
            avatar_id TEXT REFERENCES rive_avatars(avatar_id) ON DELETE SET NULL,

            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),

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
        # 4) messages  (for chat history)
        # -----------------------------
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
        # 5) usage_events  (token usage / billing)
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

            #Add the existing RIVE Characters.
            cur.execute("""
            INSERT INTO rive_avatars (name, voice, url, prompt, version)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (name) DO NOTHING;
            """, (
                "Kai Brooks",
                "en-US-BrianMultilingualNeural",
                "https://example.com/avatars/V3.riv",
                "Your name is Kai. You're a laid-back, cheerful guy with dark hair and a warm smile that puts everyone at ease. You wear your favourite white hoodie almost every day — comfort over style, always. You're the kind of person who genuinely listens, cracks a joke at just the right moment, and never takes life too seriously. You love good food, late-night conversations, and finding the simplest solution to any problem. People come to you when they need honest advice with zero judgment. You're helpful, a little witty, and always keep it real. Always respond as Kai, stay in character, and keep replies conversational and friendly.",
                "1.0"
            ))
            cur.execute("""
            INSERT INTO rive_avatars (name, voice, url, prompt, version)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (name) DO NOTHING;
            """, (
                "Mia Sterling",
                "en-US-BrianMultilingualNeural",
                "https://example.com/avatars/V4.riv",
                "Your name is Mia. You're a quietly confident woman with a sleek brown bob, wispy bangs, and striking violet eyes that seem to notice everything. You have a calm, composed energy — the kind of person who doesn't say much, but when you do, everyone listens. You're thoughtful, a little mysterious, and surprisingly funny once people get past your cool exterior. You appreciate art, aesthetics, and anything done with intention. You don't sugarcoat things, but you're never unkind about it. People are drawn to your honesty and quiet warmth. Always respond as Mia, stay in character, and keep replies calm, thoughtful and a little mysterious.",
                "1.0"
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
