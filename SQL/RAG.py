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
from typing import Optional 
import math
from Providers.APIContracts import ChatMessageStructure
from psycopg2.extras import RealDictCursor
from openai import AsyncOpenAI
import nltk
from nltk.tokenize import sent_tokenize 
from fastapi.concurrency import run_in_threadpool

class VectorRAGService:


    def __init__(self):
        load_dotenv() #Security

        nltk.download("punkt", quiet=True)

        #Getting dynamic Data
        self.model_embed = os.getenv("OPEN_AI_EMBEDDINGS_LOW")
        self.sql_password = os.getenv("SQL_PASSWORD")

        self.client = AsyncOpenAI() #Open AI connection

        #Set up connection to postgreSQL
        self.conn = psycopg2.connect(
            host="127.0.0.1",
            dbname = "vector_db",
            user = "postgres",
            password=self.sql_password,
            port=5433
        )
        self.conn.autocommit = True  # explicit commits
        self.oai = AsyncOpenAI()            


    #Processing Data
    def split_sentences(self, text):
        #https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
        return sent_tokenize(text)

    async def get_embeddings(self, text: str): 
        #From OpenAI

        resp = await self.oai.embeddings.create(
            model=self.model_embed, 
            input=text
        )
        return resp.data[0].embedding

    async def add_data(self, text: str, site_id: str, source: Optional[str] = None):
        # Get the last chunk index for THIS site (or site+source)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(MAX(chunk_index), -1)
                FROM embeddings
                WHERE site_id = %s;
                """,
                (site_id,)
            )
            chunk_index = cur.fetchone()[0]

            for sentence in self.split_sentences(text):
                chunk_index += 1
                embedding = await self.get_embeddings(sentence)
                cur.execute(
                    """
                    INSERT INTO embeddings (site_id, source, chunk_index, data, embedding)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (site_id, source, chunk_index, sentence, embedding)
                )
            self.conn.commit()


    def get_country(self, site_id: str) -> str:
        with self.conn.cursor() as cur:
            cur.execute("SELECT country FROM client_list WHERE site_id = %s;", (site_id,))
            row = cur.fetchone()
        return row[0] if row else None

    
    def delete_data(self, siteID: Optional[str] = None, source: Optional[str] = None):
        if not siteID and not source:
            raise ValueError("Provide site_id or source")
        #Deletes data under either the site_id or under a specific source
        with self.conn.cursor() as cur:
            if siteID and source:
                cur.execute("DELETE FROM embeddings WHERE site_id = %s AND source = %s;", (siteID, source))
            elif siteID:
                cur.execute("DELETE FROM embeddings WHERE site_id = %s;", (siteID,))
            else:
                cur.execute("DELETE FROM embeddings WHERE source = %s;", (source,))
            self.conn.commit()


    #Website Manipulation



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

