'''
Nadil Kangara Karunarathna
V2
Description: Database layer. Thread-safe connection pooling with automatic
retry on dropped connections. All public method names and signatures are
identical to V1 — drop-in replacement.

Key changes from V1:
  - ThreadedConnectionPool replaces the single shared connection. The old
    design was used concurrently from multiple threadpool threads, which
    psycopg2 connections do not support.
  - The per-call "SELECT 1" health check is gone (it doubled round-trips
    and leaked a cursor). Dropped connections are now handled by catching
    OperationalError/InterfaceError and retrying once on a fresh connection.
  - Reads now end their implicit transaction (rollback) so pooled
    connections are never returned idle-in-transaction.
  - Fixed deleteScrapedDocuments (called a non-existent method, so scraped
    chunks were never actually deleted).
  - Fixed get_default_integration (selected one column, read another).
  - Vector params are passed as strings everywhere (pgvector-safe).
  - Removed unused AsyncOpenAI clients. If anything else in the codebase
    referenced rag.client / rag.oai, re-add them there.

References:
https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
https://www.psycopg.org/docs/pool.html
'''
import os
import re
import time
import uuid
import asyncio
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool
from sentence_transformers import SentenceTransformer


_local_embedder = None


def _get_embedder():
    global _local_embedder
    if _local_embedder is None:
        print("==> Loading embedding model...", flush=True)
        _local_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("==> Embedding model ready", flush=True)
    return _local_embedder


class VectorRAGService:

    def __init__(self):
        load_dotenv()
        self._pool = pg_pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=int(os.getenv("DB_POOL_MAX", "10")),
            host=os.getenv("DB_HOST"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", 5432),
            # TCP keepalives so managed Postgres (Render/Supabase/RDS) doesn't
            # silently kill idle pooled connections.
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=3,
        )
        print("==> DB pool ready", flush=True)

    def close(self):
        try:
            self._pool.closeall()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # CORE EXECUTION HELPERS
    # -----------------------------------------------------------------------

    def _run(self, fn, commit: bool = False):
        """
        Run fn(cur) on a pooled connection. Commits on success when commit=True,
        otherwise rolls back to close the read transaction. Retries exactly once
        on a dropped/broken connection.
        """
        last_exc = None
        for attempt in (0, 1):
            conn = self._pool.getconn()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    result = fn(cur)
                if commit:
                    conn.commit()
                else:
                    conn.rollback()
                self._pool.putconn(conn)
                return result
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                last_exc = e
                try:
                    self._pool.putconn(conn, close=True)
                except Exception:
                    pass
                if attempt == 0:
                    continue
                raise
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                self._pool.putconn(conn)
                raise
        raise last_exc  # unreachable, defensive

    def _fetchone(self, sql: str, params=()):
        def fn(cur):
            cur.execute(sql, params)
            return cur.fetchone()
        return self._run(fn)

    def _fetchall(self, sql: str, params=()):
        def fn(cur):
            cur.execute(sql, params)
            return cur.fetchall()
        return self._run(fn)

    def _execute(self, sql: str, params=(), returning: bool = False):
        """Write query. Returns the RETURNING row when returning=True, else rowcount."""
        def fn(cur):
            cur.execute(sql, params)
            return cur.fetchone() if returning else cur.rowcount
        return self._run(fn, commit=True)

    # -----------------------------------------------------------------------
    # ACCOUNT METHODS
    # -----------------------------------------------------------------------

    def createAccount(self, user_id: str, email: str = None, name: str = None, phone: str = None):
        self._execute("""
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

    def getAccount(self, user_id: str) -> dict:
        row = self._fetchone("""
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
        self._execute("""
            UPDATE sessions SET summary = %s, updated_at = NOW() WHERE id = %s
        """, (summary, chat_id))

    def get_summary(self, chat_id: str):
        row = self._fetchone("SELECT summary FROM sessions WHERE id = %s", (chat_id,))
        return row.get("summary") if row else None

    def get_avatar(self, user_id, chat_id):
        row = self._fetchone("""
            SELECT a.name AS rive_avatar, a.voice AS avatar_voice,
                   s.welcome_message, a.url AS rive_url, a.prompt AS rive_prompt
            FROM sessions s
            LEFT JOIN rive_avatars a ON s.avatar_id = a.avatar_id
            WHERE s.user_id = %s AND s.id = %s
        """, (user_id, chat_id))
        if not row:
            return None
        return (
            row.get("rive_avatar"),
            row.get("avatar_voice"),
            row.get("welcome_message"),
            row.get("rive_url"),
            row.get("rive_prompt"),
        )

    def get_session_history(self, user_id):
        return self._fetchall("""
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

    def initial_settings(self, user_id):
        pass

    def get_history(self, user_id: str, chat_id: str):
        return self._fetchall("""
            SELECT m.role, m.content, m.created_at
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE s.user_id = %s AND s.id = %s
            ORDER BY m.created_at ASC
        """, (user_id, chat_id))

    def create_session(self, user_id: str, title: str | None = None, avatar_name: str | None = None) -> str:
        chat_id = str(uuid.uuid4())

        def fn(cur):
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

        self._run(fn, commit=True)
        return chat_id

    def delete_session(self, user_id: str, chat_id: str) -> bool:
        deleted = self._execute(
            "DELETE FROM sessions WHERE id = %s AND user_id = %s",
            (chat_id, user_id),
        )
        return deleted == 1

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        self._execute("""
            INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)
        """, (chat_id, role, content))

    def update_last_message(self, chat_id: str, last_message: str, title: str | None = None) -> None:
        if title is not None:
            self._execute("""
                UPDATE sessions
                SET last_message = %s, title = %s, updated_at = NOW()
                WHERE id = %s
            """, (last_message, title, chat_id))
        else:
            self._execute("""
                UPDATE sessions
                SET last_message = %s, updated_at = NOW()
                WHERE id = %s
            """, (last_message, chat_id))

    def get_recent_messages(self, user_id: str, chat_id: str, limit: int = 20):
        rows = self._fetchall("""
            SELECT m.role, m.content
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE s.user_id = %s AND s.id = %s
            ORDER BY m.created_at DESC
            LIMIT %s
        """, (user_id, chat_id, limit)) or []
        return list(reversed(rows))

    # -----------------------------------------------------------------------
    # COST TRACKING
    # -----------------------------------------------------------------------

    def updateCurrentTokens(self, user_id: str, amount: float):
        self._execute("""
            UPDATE accounts SET monthly_token_used = monthly_token_used + %s
            WHERE user_id = %s
        """, (amount, user_id))

    def getTokens(self, user_id: str) -> float:
        row = self._fetchone("SELECT monthly_token_used FROM accounts WHERE user_id = %s", (user_id,))
        return row["monthly_token_used"] if row else 0.0

    def getLimitTokens(self, user_id: str) -> float:
        row = self._fetchone("SELECT monthly_token_limit FROM accounts WHERE user_id = %s", (user_id,))
        return row["monthly_token_limit"] if row else 0.0

    def updateCurrentCost(self, user_id: str, amount: float):
        self.updateCurrentTokens(user_id, amount)

    def getCurrentCost(self, user_id: str) -> float:
        return self.getTokens(user_id)

    def getCostLimit(self, user_id: str) -> float:
        return self.getLimitTokens(user_id)

    def resetMonthlyCost(self, user_id: str):
        self._execute("""
            UPDATE accounts SET monthly_token_used = 0, updated_at = NOW()
            WHERE user_id = %s
        """, (user_id,))

    # -----------------------------------------------------------------------
    # BILLING CYCLE
    # -----------------------------------------------------------------------

    def checkBillingCycleReset(self, user_id: str) -> bool:
        row = self._fetchone("""
            SELECT (billing_cycle_start IS NOT NULL
                    AND billing_cycle_start <= NOW() - INTERVAL '30 days') AS cycle_expired
            FROM accounts WHERE user_id = %s
        """, (user_id,))
        return bool(row["cycle_expired"]) if row else False

    def checkBillingCycleExpired(self, user_id: str) -> bool:
        return self.checkBillingCycleReset(user_id)

    def resetBillingCycle(self, user_id: str):
        self._execute("""
            UPDATE accounts SET monthly_token_used = 0, billing_cycle_start = NOW()
            WHERE user_id = %s
        """, (user_id,))

    def grantFreeDailyCredit(self, user_id: str):
        self._execute("""
            UPDATE accounts
            SET credits_remaining = 1, billing_cycle_start = NOW()
            WHERE user_id = %s
              AND is_subscribed = FALSE
              AND credits_remaining < 1
              AND (billing_cycle_start IS NULL
                   OR billing_cycle_start <= NOW() - INTERVAL '24 hours')
        """, (user_id,))

    # -----------------------------------------------------------------------
    # STRIPE
    # -----------------------------------------------------------------------

    def getStripeCustomerId(self, user_id: str) -> str | None:
        row = self._fetchone("SELECT stripe_customer_id FROM accounts WHERE user_id = %s", (user_id,))
        return row["stripe_customer_id"] if row else None

    def setStripeCustomerId(self, user_id: str, stripe_customer_id: str):
        self._execute("""
            UPDATE accounts SET stripe_customer_id = %s, updated_at = NOW()
            WHERE user_id = %s
        """, (stripe_customer_id, user_id))

    def getUserIdByStripeCustomerId(self, stripe_customer_id: str) -> str | None:
        row = self._fetchone("SELECT user_id FROM accounts WHERE stripe_customer_id = %s", (stripe_customer_id,))
        return row["user_id"] if row else None

    def getUserEmail(self, user_id: str) -> str | None:
        row = self._fetchone("SELECT email FROM accounts WHERE user_id = %s", (user_id,))
        return row["email"] if row else None

    def setSubscriptionActive(self, user_id: str, is_active: bool):
        self._execute("""
            UPDATE accounts
            SET is_subscribed = %s, subscription_status = %s, updated_at = NOW()
            WHERE user_id = %s
        """, (is_active, "active" if is_active else "cancelled", user_id))

    def getSubscriptionStatus(self, user_id: str) -> bool:
        row = self._fetchone("SELECT is_subscribed FROM accounts WHERE user_id = %s", (user_id,))
        return bool(row["is_subscribed"]) if row else False

    # -----------------------------------------------------------------------
    # CREDITS
    # -----------------------------------------------------------------------

    def getCreditsRemaining(self, user_id: str) -> float:
        row = self._fetchone("SELECT credits_remaining FROM accounts WHERE user_id = %s", (user_id,))
        return row["credits_remaining"] if row else 0

    def addCredits(self, user_id: str, amount: float):
        self._execute("""
            UPDATE accounts
            SET credits_remaining = credits_remaining + %s, updated_at = NOW()
            WHERE user_id = %s
        """, (amount, user_id))

    def resetCredits(self, user_id: str, amount: float):
        self._execute("""
            UPDATE accounts SET credits_remaining = %s, updated_at = NOW()
            WHERE user_id = %s
        """, (amount, user_id))

    def deductCredits(self, user_id: str, credits_used: float):
        row = self._execute("""
            UPDATE accounts
            SET credits_remaining = GREATEST(credits_remaining - %s, 0),
                updated_at = NOW()
            WHERE user_id = %s
            RETURNING credits_remaining
        """, (credits_used, user_id), returning=True)
        return row["credits_remaining"] if row else 0

    def reserveCredits(self, user_id: str, amount: float) -> bool:
        row = self._execute("""
            UPDATE accounts
            SET credits_remaining = credits_remaining - %s,
                credits_reserved = COALESCE(credits_reserved, 0) + %s,
                updated_at = NOW()
            WHERE user_id = %s
              AND credits_remaining >= %s
            RETURNING credits_remaining
        """, (amount, amount, user_id, amount), returning=True)
        return row is not None

    def settleCredits(self, user_id: str, reserved: float, actual: float):
        difference = reserved - actual
        self._execute("""
            UPDATE accounts
            SET credits_remaining = credits_remaining + %s,
                credits_reserved = GREATEST(COALESCE(credits_reserved, 0) - %s, 0),
                updated_at = NOW()
            WHERE user_id = %s
        """, (difference, reserved, user_id))

    def releaseReservedCredits(self, user_id: str, amount: float):
        self._execute("""
            UPDATE accounts
            SET credits_remaining = credits_remaining + %s,
                credits_reserved = GREATEST(COALESCE(credits_reserved, 0) - %s, 0),
                updated_at = NOW()
            WHERE user_id = %s
        """, (amount, amount, user_id))

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
        business_description: str = None,
        personality_on: bool = None,
    ):
        self._execute("""
            INSERT INTO api_keys (
                key, owner_user_id, business_name,
                avatar_name, system_prompt, monthly_limit,
                business_description, personality_on
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (key, owner_user_id, business_name, avatar_name, system_prompt,
              monthly_limit, business_description, personality_on))

    def getApiKey(self, key: str) -> dict | None:
        row = self._fetchone("SELECT * FROM api_keys WHERE key = %s", (key,))
        return dict(row) if row else None

    def updateApiKey(self, key, business_name=None, avatar_name=None,
                     system_prompt=None, monthly_limit=None, is_active=None,
                     business_description=None, website_url=None,
                     last_scrape_at=None, assistant_name=None,
                     assistant_version=None, outer_color=None,
                     message_color=None, user_message_color=None,
                     font_color=None, icon_size=None, font=None,
                     font_size=None, welcome_message=None,
                     popup_questions=None):
        self._execute("""
            UPDATE api_keys SET
                business_name = COALESCE(%s, business_name),
                avatar_name = COALESCE(%s, avatar_name),
                system_prompt = COALESCE(%s, system_prompt),
                monthly_limit = COALESCE(%s, monthly_limit),
                is_active = COALESCE(%s, is_active),
                business_description = COALESCE(%s, business_description),
                website_url = COALESCE(%s, website_url),
                last_scrape_at = COALESCE(%s, last_scrape_at),
                assistant_name = COALESCE(%s, assistant_name),
                assistant_version = COALESCE(%s, assistant_version),
                outer_color = COALESCE(%s, outer_color),
                message_color = COALESCE(%s, message_color),
                user_message_color = COALESCE(%s, user_message_color),
                font_color = COALESCE(%s, font_color),
                icon_size = COALESCE(%s, icon_size),
                font = COALESCE(%s, font),
                font_size = COALESCE(%s, font_size),
                welcome_message = COALESCE(%s, welcome_message),
                popup_questions = COALESCE(%s, popup_questions),
                updated_at = NOW()
            WHERE key = %s
        """, (business_name, avatar_name, system_prompt, monthly_limit,
              is_active, business_description, website_url, last_scrape_at,
              assistant_name, assistant_version, outer_color, message_color,
              user_message_color, font_color, icon_size, font, font_size,
              welcome_message, popup_questions, key))

    def deleteApiKey(self, key: str):
        self._execute("DELETE FROM api_keys WHERE key = %s", (key,))

    def incrementConversationCount(self, key: str):
        self._execute("""
            UPDATE api_keys
            SET conversations_used = conversations_used + 1, updated_at = NOW()
            WHERE key = %s
        """, (key,))

    def resetConversationCounts(self):
        self._execute("UPDATE api_keys SET conversations_used = 0, updated_at = NOW()")

    def listApiKeys(self, owner_user_id: str) -> list:
        rows = self._fetchall("""
            SELECT key, business_name, avatar_name, monthly_limit,
                   conversations_used, is_active, created_at
            FROM api_keys
            WHERE owner_user_id = %s
            ORDER BY created_at DESC
        """, (owner_user_id,))
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # AVATAR LOOKUP
    # -----------------------------------------------------------------------

    def getAvatarByName(self, name: str) -> dict | None:
        row = self._fetchone("SELECT * FROM rive_avatars WHERE name = %s", (name,))
        return dict(row) if row else None

    # -----------------------------------------------------------------------
    # EMBEDDING / RAG
    # -----------------------------------------------------------------------

    async def embedText(self, text: str) -> list[float]:
        return await run_in_threadpool(
            lambda: _get_embedder().encode(text, convert_to_numpy=True).tolist()
        )

    def storeDocumentChunk(
        self,
        doc_id: str,
        api_key: str,
        chunk_index: int,
        content: str,
        embedding: list[float],
        filename: str = None,
    ):
        self._execute("""
            INSERT INTO document_chunks (
                doc_id, api_key, chunk_index, content, embedding, filename
            )
            VALUES (%s, %s, %s, %s, %s::vector, %s)
        """, (doc_id, api_key, chunk_index, content, str(embedding), filename))

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

        t1 = time.perf_counter()
        rows = await run_in_threadpool(
            self.searchDocumentChunks, api_key, embedding, num_results
        )
        t_sql = time.perf_counter() - t1
        print(f"==> RAG local: {t_embed:.3f}s  sql: {t_sql:.3f}s  results: {len(rows)}")

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
        """Synchronous RAG retrieval. Vector passed as string (pgvector-safe)."""
        vec = str(embedding)
        rows = self._fetchall("""
            SELECT content, filename, chunk_index,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            WHERE api_key = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (vec, api_key, vec, limit))
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # DOCUMENT MANAGEMENT
    # -----------------------------------------------------------------------

    def listDocuments(self, api_key: str) -> list[dict]:
        rows = self._fetchall("""
            SELECT DISTINCT doc_id, filename,
                COUNT(*) as chunks,
                MIN(created_at) as uploaded_at
            FROM document_chunks
            WHERE api_key = %s
            GROUP BY doc_id, filename
            ORDER BY uploaded_at DESC
        """, (api_key,))
        return [dict(r) for r in rows]

    def deleteDocument(self, doc_id: str):
        self._execute("DELETE FROM document_chunks WHERE doc_id = %s", (doc_id,))

    def deleteAllDocuments(self, api_key: str):
        self._execute("DELETE FROM document_chunks WHERE api_key = %s", (api_key,))

    def deleteScrapedDocuments(self, api_key: str) -> int:
        """
        Deletes all document chunks that came from a website scrape for this
        api_key (filename starts with "website:"). Used before re-scraping.
        Fixed in V2 — previously called a non-existent method and silently
        deleted nothing.
        """
        try:
            return self._execute("""
                DELETE FROM document_chunks
                WHERE api_key = %s AND filename LIKE 'website:%%'
            """, (api_key,))
        except Exception as e:
            print(f"==> deleteScrapedDocuments failed: {e}")
            return 0

    def hasDocuments(self, api_key: str) -> bool:
        try:
            row = self._fetchone(
                "SELECT EXISTS(SELECT 1 FROM document_chunks WHERE api_key = %s LIMIT 1) AS ok",
                (api_key,),
            )
            return bool(row["ok"]) if row else False
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # BUSINESS CREDITS
    # -----------------------------------------------------------------------

    def getBusinessCredits(self, user_id: str) -> float:
        row = self._fetchone("SELECT business_credits FROM accounts WHERE user_id = %s", (user_id,))
        return float(row["business_credits"]) if row else 0.0

    def deductBusinessCredits(self, user_id: str, amount: float) -> float:
        row = self._execute("""
            UPDATE accounts
            SET business_credits = GREATEST(business_credits - %s, 0),
                updated_at = NOW()
            WHERE user_id = %s
            RETURNING business_credits
        """, (amount, user_id), returning=True)
        return float(row["business_credits"]) if row else 0.0

    def addBusinessCredits(self, user_id: str, amount: float):
        self._execute("""
            UPDATE accounts
            SET business_credits = business_credits + %s, updated_at = NOW()
            WHERE user_id = %s
        """, (amount, user_id))

    def hasEnoughBusinessCredits(self, user_id: str, min_credits: float = 0.01) -> bool:
        return self.getBusinessCredits(user_id) >= min_credits

    # -----------------------------------------------------------------------
    # TOGGLES
    # -----------------------------------------------------------------------

    def toggle_diagram_usage(self, user_id: str, value: bool):
        self._execute("UPDATE accounts SET diagram_use = %s WHERE user_id = %s", (value, user_id))

    def get_diagram_usage(self, user_id: str) -> bool:
        row = self._fetchone("SELECT diagram_use FROM accounts WHERE user_id = %s", (user_id,))
        return bool(row["diagram_use"]) if row else True

    def toggle_pro_usage(self, user_id: str, value: bool):
        self._execute("UPDATE accounts SET pro_use = %s WHERE user_id = %s", (value, user_id))

    def get_pro_usage(self, user_id: str) -> bool:
        row = self._fetchone("SELECT pro_use FROM accounts WHERE user_id = %s", (user_id,))
        return bool(row["pro_use"]) if row else False

    def set_model(self, user_id: str, value: str):
        self._execute("UPDATE accounts SET model = %s WHERE user_id = %s", (value, user_id))

    def get_model(self, user_id: str):
        row = self._fetchone("SELECT model FROM accounts WHERE user_id = %s", (user_id,))
        return row["model"] if row else "gemini"

    # -----------------------------------------------------------------------
    # VISUAL HISTORY
    # -----------------------------------------------------------------------

    def save_visual(self, session_id: str, visual_type: str, content: str, language: str = None):
        self._execute("""
            INSERT INTO message_visuals (session_id, visual_type, content, language)
            VALUES (%s, %s, %s, %s)
        """, (session_id, visual_type, content, language))

    def get_visuals(self, session_id: str, limit: int = 20):
        return self._fetchall("""
            SELECT visual_type, content, language, created_at
            FROM message_visuals
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (session_id, limit))

    # -----------------------------------------------------------------------
    # EMBED SESSIONS
    # -----------------------------------------------------------------------

    def get_or_create_embed_session(self, session_id: str, api_key: str, owner_user_id: str) -> str:
        def fn(cur):
            cur.execute("SELECT id FROM sessions WHERE id = %s", (session_id,))
            if cur.fetchone():
                return session_id
            cur.execute("""
                INSERT INTO sessions (id, user_id, created_at, updated_at)
                VALUES (%s, %s, NOW(), NOW())
                ON CONFLICT (id) DO NOTHING
            """, (session_id, owner_user_id))
            return session_id

        return self._run(fn, commit=True)

    # -----------------------------------------------------------------------
    # MICROSOFT OAUTH TOKENS
    # -----------------------------------------------------------------------

    def save_microsoft_tokens(self, user_id: str, access_token: str, refresh_token: str, expires_at, scopes: str = None):
        self._execute("""
            INSERT INTO microsoft_tokens (user_id, access_token, refresh_token, token_expires_at, scopes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                access_token = EXCLUDED.access_token,
                refresh_token = EXCLUDED.refresh_token,
                token_expires_at = EXCLUDED.token_expires_at,
                scopes = EXCLUDED.scopes,
                updated_at = NOW()
        """, (user_id, access_token, refresh_token, expires_at, scopes))

    def get_microsoft_tokens(self, user_id: str) -> dict | None:
        row = self._fetchone("""
            SELECT access_token, refresh_token, token_expires_at, scopes
            FROM microsoft_tokens WHERE user_id = %s
        """, (user_id,))
        return dict(row) if row else None

    def delete_microsoft_tokens(self, user_id: str):
        self._execute("DELETE FROM microsoft_tokens WHERE user_id = %s", (user_id,))

    # -----------------------------------------------------------------------
    # GOOGLE OAUTH TOKENS
    # -----------------------------------------------------------------------

    def save_google_tokens(self, user_id: str, access_token: str, refresh_token: str, expires_at, scopes: str = None):
        self._execute("""
            INSERT INTO google_tokens (user_id, access_token, refresh_token, token_expires_at, scopes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                access_token = EXCLUDED.access_token,
                refresh_token = EXCLUDED.refresh_token,
                token_expires_at = EXCLUDED.token_expires_at,
                scopes = EXCLUDED.scopes,
                updated_at = NOW()
        """, (user_id, access_token, refresh_token, expires_at, scopes))

    def get_google_tokens(self, user_id: str) -> dict | None:
        row = self._fetchone("""
            SELECT access_token, refresh_token, token_expires_at, scopes
            FROM google_tokens WHERE user_id = %s
        """, (user_id,))
        return dict(row) if row else None

    def delete_google_tokens(self, user_id: str):
        self._execute("DELETE FROM google_tokens WHERE user_id = %s", (user_id,))

    # -----------------------------------------------------------------------
    # INTEGRATIONS
    # -----------------------------------------------------------------------

    def checkIntegrations(self, user_id: str):
        row = self._fetchone("SELECT integrator FROM accounts WHERE user_id = %s", (user_id,))
        return row["integrator"] if row else None

    def getEmbedDiagrams(self, owner_user_id: str) -> bool:
        row = self._fetchone("SELECT embed_diagrams FROM accounts WHERE user_id = %s", (owner_user_id,))
        return bool(row.get("embed_diagrams")) if row else False

    def getDefaultIntegration(self, user_id: str) -> str | None:
        row = self._fetchone("SELECT default_integration FROM accounts WHERE user_id = %s", (user_id,))
        return row["default_integration"] if row else None

    def set_integrator_active(self, user_id: str, active: bool):
        self._execute("UPDATE accounts SET integrator = %s WHERE user_id = %s", (active, user_id))

    def set_default_integration(self, user_id: str, provider: str | None):
        self._execute("UPDATE accounts SET default_integration = %s WHERE user_id = %s", (provider, user_id))

    def get_default_integration(self, user_id: str) -> str | None:
        # Fixed in V2: previously selected business_integration but read
        # default_integration (KeyError whenever a row existed).
        row = self._fetchone("SELECT default_integration FROM accounts WHERE user_id = %s", (user_id,))
        return row["default_integration"] if row else None

    # -----------------------------------------------------------------------
    # KNOWLEDGE GRAPH / AVAILABILITY / CALENDAR FLAGS
    # -----------------------------------------------------------------------

    def saveKnowledgeGraph(self, key: str, knowledge_graph: str):
        self._execute(
            "UPDATE api_keys SET knowledge_graph = %s, updated_at = NOW() WHERE key = %s",
            (knowledge_graph, key),
        )

    def saveAvailability(self, key: str, availability: str):
        self._execute(
            "UPDATE api_keys SET availability = %s, updated_at = NOW() WHERE key = %s",
            (availability, key),
        )

    def getCalendarEnabled(self, key: str) -> bool:
        try:
            row = self._fetchone("SELECT calendar_enabled FROM api_keys WHERE key = %s", (key,))
            return bool(row["calendar_enabled"]) if row else False
        except Exception:
            return False

    def setCalendarEnabled(self, key: str, value: bool):
        self._execute(
            "UPDATE api_keys SET calendar_enabled = %s, updated_at = NOW() WHERE key = %s",
            (value, key),
        )

    def addApiKeyCost(self, key: str, cost: float):
        """Accumulate AUD cost against the api_key row for per-deployment billing."""
        self._execute("""
            UPDATE api_keys
            SET total_cost = COALESCE(total_cost, 0) + %s, updated_at = NOW()
            WHERE key = %s
        """, (cost, key))