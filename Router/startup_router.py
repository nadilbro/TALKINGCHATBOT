from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import asyncio
import time
from Providers.APIContracts import ChatBotEdits, AccountInit, UserID
from SQL.RAG import VectorRAGService
from Providers.ai_provider import AIProvider
from Providers.startup_provider import StartUp
import os
from fastapi import UploadFile, File, Form
import tempfile
import tempfile
from typing import Optional 
import hmac, hashlib, base64
from fastapi import Request, HTTPException
from urllib.parse import urlparse
from fastapi import Query

router = APIRouter(prefix="/startup", tags=["startup"])

rag = VectorRAGService()
ai = AIProvider(rag)
startup = StartUp()




# @router.put("/create_client")
# def create_client(details: AccountInit):
#     user_id = startup.init_site_id()
#     return rag.initialise_client(user_id, details)


@router.get("/initialise_session_history")
async def initialise_session_history(user_id: str = Query(...)):
    return rag.get_history(user_id) 

@router.get("/initialise_settings")
def initalise_settings(user_id: str = Query(...)):
    return rag.initial_settings(user_id)