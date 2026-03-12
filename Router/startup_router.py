from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import asyncio
import time
from Providers.APIContracts import ChatBotEdits, AccountInit, UserID, SessionInit
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

'''CONTRACTS'''
from Providers.APIContracts import ChatBotEdits, AccountInit, UserID, SessionInit, SessionCreate

router = APIRouter(prefix="/startup", tags=["startup"])

rag = VectorRAGService()
ai = AIProvider(rag)
startup = StartUp()



@router.get("/initialise_sessions")
#This is used to initialise the sessions so upon the user login, the old sessions can load.
#QUESTION: Maybe enter a max input for this
async def initialise_sessions(user_id: str = Query(...)):
    return rag.get_session_history(user_id) 

@router.post("/create_session")
#This is used to create a new session to be registered into the database
async def create_session_route(data: SessionCreate):
    chat_id = rag.create_session(user_id=data.user_id, title=data.title, )
    return {"chat_id": chat_id}



# @router.get("/initialise_settings")
# #This is any like intial settings the user may have custom. THOUGH FOR NOW WE ARE NOT USING IT 
# def initalise_settings(user_id: str = Query(...)):
#     return rag.initial_settings(user_id)