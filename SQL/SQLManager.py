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
from dotenv import load_dotenv
import psycopg2
from typing import Optional, List, Dict, Any
from Providers.APIContracts import ChatMessageStructure, ChatBotEdits, ClientListSetUp
from psycopg2.extras import RealDictCursor
from openai import AsyncOpenAI
import datetime
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
                avatar_id = None
                if avatar_name:
                    cur.execute("""
                        SELECT avatar_id FROM rive_avatars WHERE name = %s
                    """, (avatar_name,))
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
                cur.execute("""
                    DELETE FROM sessions WHERE id = %s AND user_id = %s
                """, (chat_id, user_id))
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
        """
        Grants 1 free credit if:
        - User is not subscribed
        - credits_remaining < 1
        - 24 hours have passed since last grant
        """
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
                        monthly_token_used = monthly_token_used + %s,
                        updated_at = NOW()
                    WHERE user_id = %s
                    RETURNING credits_remaining
                """, (credits_used, credits_used, user_id))
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