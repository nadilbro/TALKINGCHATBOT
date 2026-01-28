from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import asyncio
import time
from Providers.APIContracts import StatusResponse, ChatBotEdits, SiteID, ClientListSetUp
from SQL.RAG import VectorRAGService
from Providers.ai_provider import AIProvider
from Providers.startup_provider import StartUp
import os
from fastapi import UploadFile, File, Form
import tempfile
import tempfile
from typing import Optional 


router = APIRouter(prefix="/startup", tags=["startup"])

rag = VectorRAGService()
ai = AIProvider(rag)
startup = StartUp()

@router.put("/create_client")
def create_client(details: ClientListSetUp):
    site_id = startup.init_site_id()
    return rag.initialise_client(site_id, details)

@router.put("/get_site_id")
def get_siteid_wo_client(firebase_id: str = Query(...)):
    return rag.get_siteid_wo_client(firebase_id)

