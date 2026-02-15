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
from Providers.APIContracts import ChatMessageStructure, SiteID, ChatBotEdits, AddDataRequest, EmbeddingRow, GetDataRequest, ClientListSetUp
from psycopg2.extras import RealDictCursor
from openai import AsyncOpenAI
import nltk
from fastapi.concurrency import run_in_threadpool
import datetime
from psycopg2 import sql
import re
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


    '''-------------------------EMBEDDINGS------------------------------'''
    async def get_embeddings(self, text: str): 
        #From OpenAI

        resp = await self.oai.embeddings.create(
            model=self.model_embed, 
            input=text
        )
        return resp.data[0].embedding

    async def add_data(self, info: AddDataRequest):
        # Get the last chunk index for THIS site (or site+source)
        
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(MAX(chunk_index), -1)
                FROM embeddings
                WHERE site_id = %s;
                """,
                (info.site_id,)
            )
            chunk_index = cur.fetchone()[0]

            for sentence in self.split_sentences(info.text):
                chunk_index += 1
                embedding = await self.get_embeddings(sentence)
                cur.execute(
                    """
                    INSERT INTO embeddings (site_id, source, chunk_index, data, embedding)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (info.site_id, info.source, chunk_index, sentence, embedding)
                )
            self.conn.commit()




    
    def delete_data(self, siteID: Optional[str] = None, chunk_id: Optional[str] = None):
        if not siteID or not chunk_id:
            raise ValueError("Provide site_id or source")
        #Deletes data under either the site_id or under a specific source
        with self.conn.cursor() as cur:
            if siteID and chunk_id:
                cur.execute("DELETE FROM embeddings WHERE site_id = %s AND chunk_index = %s;", (siteID, chunk_id))
            self.conn.commit()





    def get_embedding_data(self, site_id: str) -> List[EmbeddingRow]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, site_id, source, chunk_index, data, created_at
                FROM embeddings
                WHERE site_id = %s
                ORDER BY chunk_index ASC;
                """,
                (site_id,)
            )
            rows = cur.fetchall()

        # Convert datetime -> isoformat for JSON
        for r in rows:
            if r.get("created_at"):
                r["created_at"] = r["created_at"].isoformat()

        return rows
        #Website Manipulation


#--------------------------Allowed Domains-----------------------------------
    def add_allowed(self, site_id: str):
        pass

    def check_allowed(self, domain: str) -> bool:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM client_embeddings
                    WHERE domain_1 = %s
                )
                """,
                (domain,)
            )
            row = cur.fetchone()

        return bool(row["exists"])

#--------------ALLOWED WEBSITES FOR AI-----------------------------#


    def get_websites(self, site_id: str) -> list[str]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT domain
                FROM allowed_websites
                WHERE site_id = %s
                """,
                (site_id,)
            )
            rows = cur.fetchall()

        # Extract domains
        domain_list = [row["domain"] for row in rows]

        return domain_list


    def add_website(self, site_id: str, domain: str):

        with self.conn.cursor() as cur:
            # How many already exist?
            cur.execute("SELECT COUNT(*) FROM allowed_websites WHERE site_id = %s", (site_id,))
            existing_count = cur.fetchone()[0]

            remaining = max(0, 5 - existing_count)
            if remaining == 0:
                return {"status": "error", "message": "Max 5 allowed websites reached for this site."}

            # Insert up to remaining slots
            cur.execute(
                """
                INSERT INTO allowed_websites (site_id, domain)
                VALUES (%s, %s)
                ON CONFLICT (site_id, domain) DO NOTHING
                """,
                (site_id, domain)
            )
                # rowcount = 1 if inserted, 0 if duplicate

            self.conn.commit()
            return {"status": "ok", "message": f"Added {domain} website"}

    def delete_website(self, siteID: str, website: str):
        with self.conn.cursor() as cur:
            cur.execute("""
                DELETE FROM allowed_websites
                WHERE site_id = %s AND domain = %s;
            """, (siteID, website))
        self.conn.commit()
        return {"status": "ok", "message": f"deleted {website}."}


    def get_client(self, detail: str, site_id: str):
        try:
            self.conn.rollback()  # clears any failed/open tx
        except Exception:
            pass
        allowed_columns = {
            "description",
            "name",
            "email",
            "phone",
            "address"
        }

        if detail not in allowed_columns:
            raise ValueError("Invalid detail requested")
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT description
                FROM client_list
                WHERE site_id = %s;
            """, (site_id,)
            )
            row = cur.fetchone()
        return row["description"] if row else None

    async def process_question(self, userQ: str, siteID: str, numRank: int) -> ChatMessageStructure:
        #https://medium.com/@serkan_ozal/vector-similarity-search-53ed42b951d9
        #AI helped the understanding of this 
        # Better to do in Postgres for scalability and using O(N) notation, it isnt efficient
        t0 = time.perf_counter()
        embedReq = await self.get_embeddings(userQ)
        t_embed = time.perf_counter() - t0

        def _db_search():
            t1 = time.perf_counter()
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT data,
                        1 - (embedding <=> (%s)::vector) AS similarity,
                        source,
                        chunk_index
                    FROM embeddings
                    WHERE site_id = %s
                    ORDER BY embedding <=> (%s)::vector
                    LIMIT %s;
                    """,
                    (embedReq, siteID, embedReq, numRank)
                )
                retrievedData = cur.fetchall()
            t_SQL  = time.perf_counter() - t1
            return retrievedData, t_SQL

        retrievedData, t_sql = await run_in_threadpool(_db_search)
        print("embed:", t_embed, "sql:", t_sql)
            # best similarity (because you ordered by distance ascending)
        best_similarity = retrievedData[0]["similarity"] if retrievedData else 0.0
        context_text = "\n".join(f"- {r['data']}" for r in retrievedData) if retrievedData else "(No relevant context found.)"
        question = ChatMessageStructure(context=context_text, userQ=userQ)

        return question, best_similarity
    
    def close(self):
        self.conn.close()

    """--------------------- Editing Chatbot traits and appearance (called in Router -> edit.py)-------------------------------"""
    def edit_traits(self, traits: ChatBotEdits):
        sql = """
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
            border_radius,
            updated_at
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
        )
        ON CONFLICT (site_id) DO UPDATE SET
            chatbot_name  = COALESCE(EXCLUDED.chatbot_name,  chatbot_settings.chatbot_name),
            personality   = COALESCE(EXCLUDED.personality,   chatbot_settings.personality),
            tone          = COALESCE(EXCLUDED.tone,          chatbot_settings.tone),
            resp_length   = COALESCE(EXCLUDED.resp_length,   chatbot_settings.resp_length),
            temperature   = COALESCE(EXCLUDED.temperature,   chatbot_settings.temperature),
            greeting      = COALESCE(EXCLUDED.greeting,      chatbot_settings.greeting),
            fallback      = COALESCE(EXCLUDED.fallback,      chatbot_settings.fallback),
            widget_color  = COALESCE(EXCLUDED.widget_color,  chatbot_settings.widget_color),
            widget_size   = COALESCE(EXCLUDED.widget_size,   chatbot_settings.widget_size),
            border_radius = COALESCE(EXCLUDED.border_radius, chatbot_settings.border_radius),
            updated_at    = NOW();
        """

        with self.conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    traits.site_id,
                    traits.chatbot_name,
                    traits.personality,
                    traits.tone,
                    traits.resp_length,
                    traits.temperature,
                    traits.greeting,
                    traits.fallback,
                    traits.widget_color,
                    traits.widget_size,
                    traits.border_radius,
                ),
            )

        self.conn.commit()
        return {"status": "ok", "message": "Updated traits"}



    def edit_appearence(self, appearance: ChatBotEdits):
        sql = """
        INSERT INTO chatbot_settings (site_id, widget_color, widget_size, border_radius, updated_at)
        VALUES (%s, %s, %s, %s, NOW())
        ON CONFLICT (site_id) DO UPDATE SET
        widget_color  = COALESCE(EXCLUDED.widget_color,  chatbot_settings.widget_color),
        widget_size   = COALESCE(EXCLUDED.widget_size,   chatbot_settings.widget_size),
        border_radius = COALESCE(EXCLUDED.border_radius, chatbot_settings.border_radius),
        updated_at    = NOW();
        """
        ...
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                appearance.site_id,
                appearance.widget_color,
                appearance.widget_size,
                appearance.border_radius,
            ))
        self.conn.commit()
        return {"status": "ok", "message": f"Updated appearance"}
    
    

    def get_appearence(self, req: SiteID) -> ChatBotEdits:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
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
                    border_radius,
                    updated_at
                FROM chatbot_settings
                WHERE site_id = %s
                """,
                (req.site_id,)
            )
            row = cur.fetchone()

        if not row:
            # Return empty object with site_id only
            return ChatBotEdits(site_id=req.site_id)
        print(row)
        return ChatBotEdits(
            site_id=row["site_id"],
            chatbot_name=row.get("chatbot_name"),
            personality=row.get("personality"),
            tone=row.get("tone"),
            resp_length=row.get("resp_length"),
            temperature=row.get("temperature"),
            greeting=row.get("greeting"),
            fallback=row.get("fallback"),
            widget_color=row.get("widget_color"),
            widget_size=row.get("widget_size"),
            border_radius=row.get("border_radius"),
            updated_at=row["updated_at"].isoformat() if row.get("updated_at") else None
        )
    
    def get_country(self, site_id: str) -> str:
        with self.conn.cursor() as cur:
            cur.execute("SELECT country FROM client_list WHERE site_id = %s;", (site_id,))
            row = cur.fetchone()
        return row[0] if row else None
    '''---------------------------------------INITALISING CLIENT_LIST----------------------------------'''

    def initialise_client(self, site_id: str, info: ClientListSetUp) -> Dict[str, Any]:
        """
        Create client if not exists, otherwise update.
        - Does NOT overwrite existing DB values with None.
        - Always updates updated_at.
        - Returns status + whether it inserted or updated.
        """
        sql = """
        INSERT INTO client_list (
            site_id,
            name,
            email,
            phone,
            address,
            country,
            subscription,
            account_id,
            subscription_end,
            created_at,
            updated_at
        )
        VALUES (
            %(site_id)s,
            %(name)s,
            %(email)s,
            %(phone)s,
            %(address)s,
            %(country)s,
            %(subscription)s,
            %(account_id)s,
            %(subscription_end)s,
            COALESCE(%(created_at)s::timestamptz, NOW()),
            NOW(),
            %(domain_1)s,
        )
        ON CONFLICT (site_id) DO UPDATE SET
            name             = COALESCE(EXCLUDED.name, client_list.name),
            email            = COALESCE(EXCLUDED.email, client_list.email),
            phone            = COALESCE(EXCLUDED.phone, client_list.phone),
            address          = COALESCE(EXCLUDED.address, client_list.address),
            country          = COALESCE(EXCLUDED.country, client_list.country),
            subscription     = COALESCE(EXCLUDED.subscription, client_list.subscription),
            account_id       = COALESCE(EXCLUDED.account_id, client_list.account_id),
            subscription_end = COALESCE(EXCLUDED.subscription_end, client_list.subscription_end),
            updated_at       = NOW()
        RETURNING site_id, created_at, updated_at;
        """

        params = {
            "site_id": site_id,
            "name": info.name,
            "email": info.email,
            "phone": info.phone,
            "address": info.address,
            "country": info.country,
            "subscription": info.subscription,
            "account_id": info.account_id,
            "subscription_end": info.subscription_end,
            "created_at": info.created_at,
            "domain_1": info.allowed_domain
        }

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                row = cur.fetchone()

                # Determine whether insert vs update:
                # If created_at == updated_at immediately after write, it's *likely* insert,
                # but not guaranteed. Better method is to query existence first:
                # We'll do the safe way below.
                cur.execute("SELECT 1 FROM client_list WHERE site_id = %s;", (site_id,))
                exists = cur.fetchone() is not None

            self.conn.commit()
            return {
                "status": "ok",
                "message": "Client created/updated",
                "site_id": row["site_id"] if row else site_id,
                "created_at": row["created_at"].isoformat() if row and row.get("created_at") else None,
                "updated_at": row["updated_at"].isoformat() if row and row.get("updated_at") else None,
                "exists": exists
            }

        except Exception as e:
            # IMPORTANT: clear transaction state so future queries don't fail
            try:
                self.conn.rollback()
            except Exception:
                pass
            raise


    def get_siteid_wo_client(self, id: str) -> str:
        #Returns the site_id based on firebase 
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT site_id FROM client_list 
                WHERE account_id = %s
                """,
                (id,)
            )
            row = cur.fetchone() 

        return row if row else None

    #________________----SIMPLE COMMANDs------______________________________
    

    def domain_allowed(self, site_id: str, domain: str) -> bool:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM client_embeddings
                    WHERE site_id = %s
                    AND domain_1 = %s
                ) AS exists;
                """,
                (site_id, domain)
            )
            row = cur.fetchone()

        return bool(row["exists"]) if row else False
    
    def check_exists(self, key: str, value, table: str) -> bool:
        q = sql.SQL("""
            SELECT EXISTS(
                SELECT 1
                FROM {table}
                WHERE {col} = %s
            ) AS exists;
        """).format(
            table=sql.Identifier(table),
            col=sql.Identifier(key),
        )

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, (value,))
            row = cur.fetchone()

        return bool(row["exists"]) if row else False
    
    '''__________________-VOICE CHAT COMMANDS-______________________'''
    def get_avatar(self, site_id):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT avatar_link, avatar_voice, welcome_message, primary_colour FROM avatar_list WHERE site_id = %s",
                (site_id,)  # Added comma to make it a tuple
            )
            row = cur.fetchone()
            
            # Handle cases where no record is found
            if not row:
                return (None, None, None, None)
                
            # Use dictionary keys instead of indices with RealDictCursor
            return (
            row.get("avatar_link"),
            row.get("avatar_voice"),
            row.get("welcome_message"),
            row.get("primary_colour"),
        )
    def get_voice_init(self, site_id: str):
        """
        Returns: avatar_key, voice_name, welcome_message, primary_color, rive_url

        Your DB currently stores:
          avatar_link, avatar_voice, welcome_message, primary_colour
        inside avatar_list.

        We'll map:
          avatar_key   -> avatar_link
          voice_name   -> avatar_voice
          rive_url     -> avatar_link (or "" if you later store a separate rive_url)
        """
        avatar_link, avatar_voice, welcome_message, primary_colour = self.get_avatar(site_id)

        avatar_key = avatar_link  # keep naming consistent with your frontend/backends
        voice_name = avatar_voice
        primary_color = primary_colour

        # If your "avatar_link" is actually your Rive file URL, keep it here:
        rive_url = avatar_link or ""

        return avatar_key, voice_name, welcome_message, primary_color, rive_url
