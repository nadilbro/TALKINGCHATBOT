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
import time
from fastapi import APIRouter #Allows to branch main url
import os
from pydantic import BaseModel #Used with FastAPI
from dotenv import load_dotenv
import psycopg2
from typing import Optional, List, Dict, Any
import math
from Providers.APIContracts import ChatMessageStructure, ChatBotEdits, ClientListSetUp
from psycopg2.extras import RealDictCursor
from openai import AsyncOpenAI
import nltk
from fastapi.concurrency import run_in_threadpool
import datetime
from psycopg2 import sql
import re
import uuid

print("✅ RAG.py loaded: re imported OK")


class VectorRAGService:


    def __init__(self):
        load_dotenv() #Security

     #   nltk.download("punkt", quiet=True)

        #Getting dynamic Data
        self.model_embed = os.getenv("OPEN_AI_EMBEDDINGS_LOW")
        self.sql_password = os.getenv("SQL_PASSWORD")

        self.client = AsyncOpenAI() #Open AI connection

        self.conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", 5432)
        )

        self.oai = AsyncOpenAI()            


    #Processing Data
    def split_sentences(self, text: str) -> list[str]:
        # Split on sentence-ending punctuation followed by whitespace
        parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
        return [p.strip() for p in parts if p and p.strip()]
        
    def update_summary(self, chat_id: str, summary: str):
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE sessions
                    SET summary = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (summary, chat_id)
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def get_summary(self, chat_id: str):
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT summary FROM sessions WHERE id = %s
                    """,
                    (chat_id,)
                )
                row = cur.fetchone()
                if not row:
                    return None
                return row.get("summary")
        except Exception:
            self.conn.rollback()
            raise
    def get_avatar(self, user_id, chat_id):
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
        #this gets the history of all the sessions 
        #NOT THE INDUVIDUAL
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
            SELECT
                    s.id,
                    s.title,
                    s.last_message,
                    s.status,
                    a.name AS rive_avatar,
                    a.voice AS avatar_voice,
                    s.welcome_message,
                    s.summary,
                    s.created_at,
                    s.updated_at
                FROM sessions s
                LEFT JOIN rive_avatars a ON s.avatar_id = a.avatar_id
                WHERE s.user_id = %s
                ORDER BY s.updated_at DESC
            """, (user_id,))

            rows = cur.fetchall()

            # Always return a list (empty list if no history)
            return rows

    def initial_settings(self, user_id):
        #Not in use nor called yet
        pass


    def get_history(self, user_id: str, chat_id: str):
        """
        Description: This is for when the user clicks on a session
        Returns full message history for a session
        only if the session belongs to the given user.
        Returns: List[Dict]
        
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT m.role, m.content, m.created_at
                FROM messages m
                JOIN sessions s ON m.session_id = s.id
                WHERE s.user_id = %s
                AND s.id = %s
                ORDER BY m.created_at ASC
            """, (user_id, chat_id))

            rows = cur.fetchall()

            # Always return a list (empty list if no history)
            return rows
        
    #CREATING A NEW SESSION AND SAVING INFORMATION
    def create_session(self, user_id: str, title: str | None = None, avatar_name: str | None = None) -> str:
        chat_id = str(uuid.uuid4())

        try:
            #First we check if the account exists. 
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO accounts (user_id)
                    VALUES (%s)
                    ON CONFLICT (user_id) DO NOTHING
                """, (user_id,))

                avatar_id = None
                if avatar_name:
                    cur.execute("""
                        SELECT avatar_id, voice, url, prompt FROM rive_avatars WHERE name = %s
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
    #Deleting a session and their equivelant messsages
    def delete_session(self, user_id: str, chat_id: str) -> bool:
        try:
            with self.conn.cursor() as cur:
                # Verify the session belongs to this user, then delete it
                # Messages are deleted automatically via ON DELETE CASCADE
                cur.execute("""
                    DELETE FROM sessions
                    WHERE id = %s AND user_id = %s
                """, (chat_id, user_id))

                deleted = cur.rowcount  # 1 if deleted, 0 if not found / wrong user

            self.conn.commit()
            return deleted == 1

        except Exception:
            self.conn.rollback()
            raise

    def add_message(self, chat_id: str, role: str, content: str) -> None:
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
        try:
            with self.conn.cursor() as cur:
                if title is not None:
                    cur.execute("""
                        UPDATE sessions
                        SET last_message = %s,
                            title = %s,
                            updated_at = NOW()
                        WHERE id = %s
                    """, (last_message, title, chat_id))
                else:
                    cur.execute("""
                        UPDATE sessions
                        SET last_message = %s,
                            updated_at = NOW()
                        WHERE id = %s
                    """, (last_message, chat_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def get_recent_messages(self, user_id: str, chat_id: str, limit: int = 20):
        #This function is there to get the last recent messages so they can load when the user clicks on the thing.
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

        # reverse so it's chronological (oldest -> newest)
        return list(reversed(rows))
    
    def updateCurrentTokens(self, user_id: str, amount: float):
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE accounts
                    SET monthly_token_used = monthly_token_used + %s
                    WHERE user_id = %s;
                """, (amount, user_id))  # ← was wrong order before
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def getTokens(self, user_id: str) -> float:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT monthly_token_used 
                FROM accounts 
                WHERE user_id = %s
            """, (user_id,))  # ← SQL was wrong, missing = and comma
            row = cur.fetchone()
            return row["monthly_token_used"] if row else 0.0

    def getLimitTokens(self, user_id: str) -> float:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT monthly_token_limit 
                FROM accounts 
                WHERE user_id = %s
            """, (user_id,))
            row = cur.fetchone()
            return row["monthly_token_limit"] if row else 0.0

    def checkBillingCycleReset(self, user_id: str) -> bool:
        """
        Returns True if 30 days have passed since billing_cycle_start.
        Returns False if still within the cycle or billing hasn't started.
        Does NOT update anything — caller decides what to do.
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT billing_cycle_start 
                FROM accounts 
                WHERE user_id = %s
            """, (user_id,))
            row = cur.fetchone()

            if not row or row["billing_cycle_start"] is None:
                return False  # No billing start = not a paid member, don't reset

            cur.execute("""
                SELECT billing_cycle_start <= NOW() - INTERVAL '30 days' AS cycle_expired
                FROM accounts
                WHERE user_id = %s
            """, (user_id,))
            result = cur.fetchone()
            return bool(result["cycle_expired"]) if result else False

    def resetBillingCycle(self, user_id: str):
        """
        Call this only after confirming payment/subscription is valid.
        Resets monthly_token_used and updates billing_cycle_start to now.
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE accounts
                    SET 
                        monthly_token_used = 0,
                        billing_cycle_start = NOW()
                    WHERE user_id = %s
                """, (user_id,))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    #STRIPE
    # -----------------------------------------------------------------------
    # STRIPE METHODS — paste these into VectorRAGService
    # -----------------------------------------------------------------------
    
    def getStripeCustomerId(self, user_id: str) -> str | None:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT stripe_customer_id FROM accounts WHERE user_id = %s
            """, (user_id,))
            row = cur.fetchone()
            return row["stripe_customer_id"] if row else None
    
    def setStripeCustomerId(self, user_id: str, stripe_customer_id: str):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE accounts
                    SET stripe_customer_id = %s, updated_at = NOW()
                    WHERE user_id = %s
                """, (stripe_customer_id, user_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
    
    def getUserIdByStripeCustomerId(self, stripe_customer_id: str) -> str | None:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT user_id FROM accounts WHERE stripe_customer_id = %s
            """, (stripe_customer_id,))
            row = cur.fetchone()
            return row["user_id"] if row else None
    
    def getUserEmail(self, user_id: str) -> str | None:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT email FROM accounts WHERE user_id = %s
            """, (user_id,))
            row = cur.fetchone()
            return row["email"] if row else None
    
    def setSubscriptionActive(self, user_id: str, is_active: bool):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE accounts
                    SET 
                        is_subscribed = %s,
                        subscription_status = %s,
                        updated_at = NOW()
                    WHERE user_id = %s
                """, (is_active, "active" if is_active else "cancelled", user_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
    
    # -----------------------------------------------------------------------
    # CREDITS METHODS
    # -----------------------------------------------------------------------
    
    def getCreditsRemaining(self, user_id: str) -> int:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT credits_remaining FROM accounts WHERE user_id = %s
            """, (user_id,))
            row = cur.fetchone()
            return row["credits_remaining"] if row else 0
    
    def addCredits(self, user_id: str, amount: int):
        """Adds credits on top of existing — used for top-up purchases."""
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
    
    def resetCredits(self, user_id: str, amount: int):
        """Sets credits to exact amount — used on monthly renewal."""
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
        """
        Deducts credits after a conversation.
        credits_used = conversation_duration_minutes / 7
        Returns remaining credits after deduction.
        """
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
    
    def hasEnoughCredits(self, user_id: str, min_credits: float = 0.1) -> bool:
        """
        Quick check before starting a conversation.
        Returns False if user has run out of credits.
        """
        remaining = self.getCreditsRemaining(user_id)
        return remaining >= min_credits
    
    # -----------------------------------------------------------------------
    # COST TRACKING METHODS (keep existing but rename for clarity)
    # -----------------------------------------------------------------------
    
    def updateCurrentCost(self, user_id: str, amount: float):
        """Alias for updateCurrentTokens — tracks dollar cost of API usage."""
        self.updateCurrentTokens(user_id, amount)
    
    def getCurrentCost(self, user_id: str) -> float:
        return self.getTokens(user_id)
    
    def getCostLimit(self, user_id: str) -> float:
        return self.getLimitTokens(user_id)
    
    def resetMonthlyCost(self, user_id: str):
        """Resets monthly cost tracking — called alongside resetBillingCycle."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE accounts
                    SET monthly_token_used = 0, updated_at = NOW()
                    WHERE user_id = %s
                """, (user_id,))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
    
    def checkBillingCycleExpired(self, user_id: str) -> bool:
        """Alias for checkBillingCycleReset."""
        return self.checkBillingCycleReset(user_id)