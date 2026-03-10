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

    def get_avatar(self, user_id, chat_id): 
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT rive_avatar, avatar_voice, welcome_message, rive_url
                FROM sessions
                WHERE user_id = %s AND id = %s
            """, (user_id, chat_id))
            row = cur.fetchone()

            if not row:
                return None, None, None, None

            return (
                row.get("rive_avatar"),
                row.get("avatar_voice"),
                row.get("welcome_message"),
                row.get("rive_url"),
            )

    def get_session_history(self, user_id):
        pass        

    def initial_settings(self, user_id):
        pass


    def get_history(self, user_id: str, chat_id: str):
        """
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
    def create_session(self, user_id: str, title: str | None = None) -> str:
        chat_id = str(uuid.uuid4())
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO sessions (id, user_id, title, status)
                VALUES (%s, %s, %s, 'Open')
            """, (chat_id, user_id, title))
        self.conn.commit()
        return chat_id

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO messages (session_id, role, content)
                VALUES (%s, %s, %s)
            """, (chat_id, role, content))
        self.conn.commit()

    def update_last_message(self, chat_id: str, last_message: str, title: str | None = None) -> None:
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

    def get_recent_messages(self, user_id: str, chat_id: str, limit: int = 20):
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