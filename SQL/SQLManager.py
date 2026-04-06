'''
Nadil Kangara Karunarathna
15/01/2026
V1
Description: This file manages anything to do with the database

References: 
https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences 
https://platform.openai.com/docs/pricing#embeddings
https://platform.openai.com/docs/guides/embeddings 
'''
import os
import time
from dotenv import load_dotenv
import psycopg2
from typing import Optional, List, Dict, Any
from Providers.APIContracts import ChatMessageStructure, ChatBotEdits, ClientListSetUp, TextIngestionRequest
from psycopg2.extras import RealDictCursor
from openai import AsyncOpenAI
from fastapi.concurrency import run_in_threadpool
import re
import uuid

print("✅ RAG.py loaded: re imported OK")


class VectorRAGService:

    def __init__(self):
        load_dotenv()

        self.model_embed = os.getenv("OPEN_AI_EMBEDDINGS_LOW")
        self.sql_password = os.getenv("SQL_PASSWORD")
        self.client = AsyncOpenAI()
        self.oai = AsyncOpenAI()
        self.conn = self._connect()

    def _connect(self):
        return psycopg2.connect(
            host=os.getenv("DB_HOST"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", 5432)
        )

    def _get_conn(self):
        """Returns a live connection, reconnecting if the connection dropped."""
        try:
            self.conn.cursor().execute("SELECT 1")
        except Exception:
            self.conn = self._connect()
        return self.conn

    # -----------------------------------------------------------------------
    # ACCOUNT METHODS
    # -----------------------------------------------------------------------

    def createAccount(self, user_id: str, email: str = None, name: str = None, phone: str = None):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO accounts (
                        user_id, email, name, phone,
                        credits_remaining, credits_reserved,
                        monthly_token_used, monthly_token_limit,
                        is_subscribed, subscription_status,
                        billing_cycle_start
                    )
                    VALUES (%s, %s, %s, %s, 1, 0, 0, 0, FALSE, 'free', NOW())
                    ON CONFLICT (user_id) DO UPDATE
                    SET 
                        email = COALESCE(EXCLUDED.email, accounts.email),
                        name = COALESCE(EXCLUDED.name, accounts.name),
                        phone = COALESCE(EXCLUDED.phone, accounts.phone),
                        updated_at = NOW()
                """, (user_id, email, name, phone))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def getAccount(self, user_id: str) -> dict:
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    user_id, name, email, phone,
                    subscription_status, stripe_customer_id,
                    stripe_subscription_id, monthly_token_limit,
                    monthly_token_used, billing_cycle_start,
                    credits_remaining, is_subscribed,
                    created_at, updated_at
                FROM accounts 
                WHERE user_id = %s
            """, (user_id,))
            row = cur.fetchone()
            return dict(row) if row else {}

    # -----------------------------------------------------------------------
    # PROCESSING
    # -----------------------------------------------------------------------

    def split_sentences(self, text: str) -> list[str]:
        parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
        return [p.strip() for p in parts if p and p.strip()]

    # -----------------------------------------------------------------------
    # SESSIONS
    # -----------------------------------------------------------------------

    def update_summary(self, chat_id: str, summary: str):
        self._get_conn()
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE sessions
                    SET summary = %s, updated_at = NOW()
                    WHERE id = %s
                """, (summary, chat_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def get_summary(self, chat_id: str):
        self._get_conn()
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT summary FROM sessions WHERE id = %s", (chat_id,))
                row = cur.fetchone()
                return row.get("summary") if row else None
        except Exception:
            self.conn.rollback()
            raise

    def get_avatar(self, user_id, chat_id):
        self._get_conn()
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT a.name AS rive_avatar, a.voice AS avatar_voice,
                           s.welcome_message, a.url AS rive_url, a.prompt AS rive_prompt
                    FROM sessions s
                    LEFT JOIN rive_avatars a ON s.avatar_id = a.avatar_id
                    WHERE s.user_id = %s AND s.id = %s
                """, (user_id, chat_id))
                row = cur.fetchone()
                if not row:
                    return None
                return (
                    row.get("rive_avatar"),
                    row.get("avatar_voice"),
                    row.get("welcome_message"),
                    row.get("rive_url"),
                    row.get("rive_prompt")
                )
        except Exception:
            self.conn.rollback()
            raise

    def get_session_history(self, user_id):
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    s.id, s.title, s.last_message, s.status,
                    a.name AS rive_avatar, a.voice AS avatar_voice,
                    s.welcome_message, s.summary,
                    s.created_at, s.updated_at
                FROM sessions s
                LEFT JOIN rive_avatars a ON s.avatar_id = a.avatar_id
                WHERE s.user_id = %s
                ORDER BY s.updated_at DESC
            """, (user_id,))
            return cur.fetchall()

    def initial_settings(self, user_id):
        pass

    def get_history(self, user_id: str, chat_id: str):
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT m.role, m.content, m.created_at
                FROM messages m
                JOIN sessions s ON m.session_id = s.id
                WHERE s.user_id = %s AND s.id = %s
                ORDER BY m.created_at ASC
            """, (user_id, chat_id))
            return cur.fetchall()

    def create_session(self, user_id: str, title: str | None = None, avatar_name: str | None = None) -> str:
        self._get_conn()
        chat_id = str(uuid.uuid4())
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO accounts (user_id, credits_remaining, credits_reserved,
                        monthly_token_used, monthly_token_limit, is_subscribed,
                        subscription_status, billing_cycle_start)
                    VALUES (%s, 1, 0, 0, 0, FALSE, 'free', NOW())
                    ON CONFLICT (user_id) DO NOTHING
                """, (user_id,))

                avatar_id = None
                if avatar_name:
                    cur.execute("SELECT avatar_id FROM rive_avatars WHERE name = %s", (avatar_name,))
                    row = cur.fetchone()
                    if row:
                        avatar_id = row["avatar_id"]

                cur.execute("""
                    INSERT INTO sessions (id, user_id, title, avatar_id, status)
                    VALUES (%s, %s, %s, %s, 'Open')
                """, (chat_id, user_id, title, avatar_id))

            self.conn.commit()
            return chat_id
        except Exception:
            self.conn.rollback()
            raise

    def delete_session(self, user_id: str, chat_id: str) -> bool:
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM sessions WHERE id = %s AND user_id = %s", (chat_id, user_id))
                deleted = cur.rowcount
            self.conn.commit()
            return deleted == 1
        except Exception:
            self.conn.rollback()
            raise

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO messages (session_id, role, content)
                    VALUES (%s, %s, %s)
                """, (chat_id, role, content))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def update_last_message(self, chat_id: str, last_message: str, title: str | None = None) -> None:
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                if title is not None:
                    cur.execute("""
                        UPDATE sessions
                        SET last_message = %s, title = %s, updated_at = NOW()
                        WHERE id = %s
                    """, (last_message, title, chat_id))
                else:
                    cur.execute("""
                        UPDATE sessions
                        SET last_message = %s, updated_at = NOW()
                        WHERE id = %s
                    """, (last_message, chat_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def get_recent_messages(self, user_id: str, chat_id: str, limit: int = 20):
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT m.role, m.content
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE s.user_id = %s AND s.id = %s
                ORDER BY m.created_at DESC
                LIMIT %s
            """, (user_id, chat_id, limit))
            rows = cur.fetchall() or []
        return list(reversed(rows))

    # -----------------------------------------------------------------------
    # COST TRACKING
    # -----------------------------------------------------------------------

    def updateCurrentTokens(self, user_id: str, amount: float):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE accounts
                    SET monthly_token_used = monthly_token_used + %s
                    WHERE user_id = %s
                """, (amount, user_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def getTokens(self, user_id: str) -> float:
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT monthly_token_used FROM accounts WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            return row["monthly_token_used"] if row else 0.0

    def getLimitTokens(self, user_id: str) -> float:
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT monthly_token_limit FROM accounts WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            return row["monthly_token_limit"] if row else 0.0

    def updateCurrentCost(self, user_id: str, amount: float):
        self.updateCurrentTokens(user_id, amount)

    def getCurrentCost(self, user_id: str) -> float:
        return self.getTokens(user_id)

    def getCostLimit(self, user_id: str) -> float:
        return self.getLimitTokens(user_id)

    def resetMonthlyCost(self, user_id: str):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE accounts SET monthly_token_used = 0, updated_at = NOW()
                    WHERE user_id = %s
                """, (user_id,))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -----------------------------------------------------------------------
    # BILLING CYCLE
    # -----------------------------------------------------------------------

    def checkBillingCycleReset(self, user_id: str) -> bool:
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT billing_cycle_start FROM accounts WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if not row or row["billing_cycle_start"] is None:
                return False
            cur.execute("""
                SELECT billing_cycle_start <= NOW() - INTERVAL '30 days' AS cycle_expired
                FROM accounts WHERE user_id = %s
            """, (user_id,))
            result = cur.fetchone()
            return bool(result["cycle_expired"]) if result else False

    def checkBillingCycleExpired(self, user_id: str) -> bool:
        return self.checkBillingCycleReset(user_id)

    def resetBillingCycle(self, user_id: str):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE accounts
                    SET monthly_token_used = 0, billing_cycle_start = NOW()
                    WHERE user_id = %s
                """, (user_id,))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def grantFreeDailyCredit(self, user_id: str):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE accounts
                    SET
                        credits_remaining = 1,
                        billing_cycle_start = NOW()
                    WHERE user_id = %s
                    AND is_subscribed = FALSE
                    AND credits_remaining < 1
                    AND (
                        billing_cycle_start IS NULL
                        OR billing_cycle_start <= NOW() - INTERVAL '24 hours'
                    )
                """, (user_id,))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -----------------------------------------------------------------------
    # STRIPE
    # -----------------------------------------------------------------------

    def getStripeCustomerId(self, user_id: str) -> str | None:
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT stripe_customer_id FROM accounts WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            return row["stripe_customer_id"] if row else None

    def setStripeCustomerId(self, user_id: str, stripe_customer_id: str):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE accounts SET stripe_customer_id = %s, updated_at = NOW()
                    WHERE user_id = %s
                """, (stripe_customer_id, user_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def getUserIdByStripeCustomerId(self, stripe_customer_id: str) -> str | None:
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT user_id FROM accounts WHERE stripe_customer_id = %s", (stripe_customer_id,))
            row = cur.fetchone()
            return row["user_id"] if row else None

    def getUserEmail(self, user_id: str) -> str | None:
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT email FROM accounts WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            return row["email"] if row else None

    def setSubscriptionActive(self, user_id: str, is_active: bool):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE accounts
                    SET is_subscribed = %s, subscription_status = %s, updated_at = NOW()
                    WHERE user_id = %s
                """, (is_active, "active" if is_active else "cancelled", user_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def getSubscriptionStatus(self, user_id: str) -> bool:
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT is_subscribed FROM accounts WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            return bool(row["is_subscribed"]) if row else False

    # -----------------------------------------------------------------------
    # CREDITS
    # -----------------------------------------------------------------------

    def getCreditsRemaining(self, user_id: str) -> float:
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT credits_remaining FROM accounts WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            return row["credits_remaining"] if row else 0

    def addCredits(self, user_id: str, amount: float):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE accounts
                    SET credits_remaining = credits_remaining + %s, updated_at = NOW()
                    WHERE user_id = %s
                """, (amount, user_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def resetCredits(self, user_id: str, amount: float):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE accounts
                    SET credits_remaining = %s, updated_at = NOW()
                    WHERE user_id = %s
                """, (amount, user_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def deductCredits(self, user_id: str, credits_used: float):
        self._get_conn()
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE accounts
                    SET
                        credits_remaining = GREATEST(credits_remaining - %s, 0),
                        updated_at = NOW()
                    WHERE user_id = %s
                    RETURNING credits_remaining
                """, (credits_used, user_id))
                row = cur.fetchone()
            self.conn.commit()
            return row["credits_remaining"] if row else 0
        except Exception:
            self.conn.rollback()
            raise

    def reserveCredits(self, user_id: str, amount: float) -> bool:
        self._get_conn()
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE accounts
                    SET
                        credits_remaining = credits_remaining - %s,
                        credits_reserved = COALESCE(credits_reserved, 0) + %s,
                        updated_at = NOW()
                    WHERE user_id = %s
                    AND credits_remaining >= %s
                    RETURNING credits_remaining
                """, (amount, amount, user_id, amount))
                row = cur.fetchone()
                if not row:
                    return False
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def settleCredits(self, user_id: str, reserved: float, actual: float):
        self._get_conn()
        try:
            difference = reserved - actual
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE accounts
                    SET
                        credits_remaining = credits_remaining + %s,
                        credits_reserved = GREATEST(COALESCE(credits_reserved, 0) - %s, 0),
                        updated_at = NOW()
                    WHERE user_id = %s
                """, (difference, reserved, user_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def releaseReservedCredits(self, user_id: str, amount: float):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE accounts
                    SET
                        credits_remaining = credits_remaining + %s,
                        credits_reserved = GREATEST(COALESCE(credits_reserved, 0) - %s, 0),
                        updated_at = NOW()
                    WHERE user_id = %s
                """, (amount, amount, user_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def hasEnoughCredits(self, user_id: str, min_credits: float = 0.1) -> bool:
        return self.getCreditsRemaining(user_id) >= min_credits

    # -----------------------------------------------------------------------
    # API KEY METHODS
    # -----------------------------------------------------------------------

    def createApiKey(
        self,
        key: str,
        owner_user_id: str,
        business_name: str,
        avatar_name: str = "Mia Sterling",
        system_prompt: str = None,
        monthly_limit: int = 500,
        business_description: str = None,  # ADD THIS
        personality_on : bool = None,  # ADD THIS
    ):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO api_keys (
                        key, owner_user_id, business_name,
                        avatar_name, system_prompt, monthly_limit, business_description, personality_on 
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (key, owner_user_id, business_name, avatar_name, system_prompt, monthly_limit, business_description, personality_on))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def getApiKey(self, key: str) -> dict | None:
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM api_keys WHERE key = %s", (key,))
            row = cur.fetchone()
            return dict(row) if row else None

    def listApiKeys(self, owner_user_id: str) -> list:
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT key, business_name, avatar_name, monthly_limit,
                    conversations_used, is_active, created_at
                FROM api_keys
                WHERE owner_user_id = %s
                ORDER BY created_at DESC
            """, (owner_user_id,))
            return [dict(r) for r in cur.fetchall()]
    def updateApiKey(self, key, business_name=None, avatar_name=None,
                    system_prompt=None, monthly_limit=None, is_active=None,
                    business_description=None, personality_on=None):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE api_keys SET
                        business_name = COALESCE(%s, business_name),
                        avatar_name = COALESCE(%s, avatar_name),
                        system_prompt = COALESCE(%s, system_prompt),
                        monthly_limit = COALESCE(%s, monthly_limit),
                        is_active = COALESCE(%s, is_active),
                        business_description = COALESCE(%s, business_description),
                        personality_on = COALESCE(%s, personality_on),
                        updated_at = NOW()
                    WHERE key = %s
                """, (business_name, avatar_name, system_prompt, monthly_limit, is_active, business_description, personality_on, key))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def deleteApiKey(self, key: str):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM api_keys WHERE key = %s", (key,))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def incrementConversationCount(self, key: str):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE api_keys
                    SET conversations_used = conversations_used + 1, updated_at = NOW()
                    WHERE key = %s
                """, (key,))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def resetConversationCounts(self):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("UPDATE api_keys SET conversations_used = 0, updated_at = NOW()")
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -----------------------------------------------------------------------
    # AVATAR LOOKUP
    # -----------------------------------------------------------------------

    def getAvatarByName(self, name: str) -> dict | None:
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM rive_avatars WHERE name = %s", (name,))
            row = cur.fetchone()
            return dict(row) if row else None

    # -----------------------------------------------------------------------
    # EMBEDDING
    # -----------------------------------------------------------------------

    async def embedText(self, text: str) -> list[float]:
        """Generates an embedding vector using OpenAI text-embedding-3-small."""
        response = await self.oai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    # -----------------------------------------------------------------------
    # DOCUMENT STORAGE
    # -----------------------------------------------------------------------

    def storeDocumentChunk(
        self,
        doc_id: str,
        api_key: str,
        chunk_index: int,
        content: str,
        embedding: list[float],
        filename: str = None,
    ):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO document_chunks (
                        doc_id, api_key, chunk_index, content, embedding, filename
                    )
                    VALUES (%s, %s, %s, %s, %s::vector, %s)
                """, (doc_id, api_key, chunk_index, content, embedding, filename))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -----------------------------------------------------------------------
    # RAG RETRIEVAL
    # -----------------------------------------------------------------------

    async def processEmbedQuestion(
        self,
        user_question: str,
        api_key: str,
        num_results: int = 3,
    ) -> tuple[str, float]:
        """
        Main RAG method for the embed system.
        Takes a user question, finds the most relevant document chunks,
        returns (context_text, best_similarity_score).
        """
        t0 = time.perf_counter()
        embedding = await self.embedText(user_question)
        t_embed = time.perf_counter() - t0

        def _db_search():
            t1 = time.perf_counter()
            self._get_conn()
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT content,
                           filename,
                           chunk_index,
                           1 - (embedding <=> (%s)::vector) AS similarity
                    FROM document_chunks
                    WHERE api_key = %s
                    ORDER BY embedding <=> (%s)::vector
                    LIMIT %s;
                """, (embedding, api_key, embedding, num_results))
                rows = cur.fetchall()
            t_sql = time.perf_counter() - t1
            return rows, t_sql

        rows, t_sql = await run_in_threadpool(_db_search)
        print(f"==> RAG embed: {t_embed:.3f}s  sql: {t_sql:.3f}s  results: {len(rows)}")

        if not rows:
            return "(No relevant context found in knowledge base.)", 0.0

        best_similarity = rows[0]["similarity"]
        context_text = "\n".join(f"- {r['content']}" for r in rows)

        return context_text, best_similarity

    def searchDocumentChunks(
        self,
        api_key: str,
        embedding: list[float],
        limit: int = 3,
    ) -> list[dict]:
        """Synchronous version of RAG retrieval."""
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT content, filename, chunk_index,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                WHERE api_key = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding, api_key, embedding, limit))
            return [dict(r) for r in cur.fetchall()]

    # -----------------------------------------------------------------------
    # DOCUMENT MANAGEMENT
    # -----------------------------------------------------------------------

    def listDocuments(self, api_key: str) -> list[dict]:
        self._get_conn()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT doc_id, filename,
                    COUNT(*) as chunks,
                    MIN(created_at) as uploaded_at
                FROM document_chunks
                WHERE api_key = %s
                GROUP BY doc_id, filename
                ORDER BY uploaded_at DESC
            """, (api_key,))
            return [dict(r) for r in cur.fetchall()]

    def deleteDocument(self, doc_id: str):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE doc_id = %s", (doc_id,))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def deleteAllDocuments(self, api_key: str):
        self._get_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE api_key = %s", (api_key,))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise