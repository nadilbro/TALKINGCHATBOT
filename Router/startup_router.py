from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import asyncio
import time
from Providers.APIContracts import StatusResponse, ChatBotEdits, SiteID, AccountInit, UserID
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

router = APIRouter(prefix="/startup", tags=["startup"])

rag = VectorRAGService()
ai = AIProvider(rag)
startup = StartUp()




# @router.put("/create_client")
# def create_client(details: AccountInit):
#     user_id = startup.init_site_id()
#     return rag.initialise_client(user_id, details)


@router.get("/initialise_session_history")
def initialise_session_history(user: UserID):
    return rag.get_history(user.user_id) 

@router.get("/initialise_settings")
def initalise_settings(user: UserID):
    return rag.initial_settings(user.user_id)