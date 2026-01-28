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


router = APIRouter(prefix="/startup", tags=["edit"])

rag = VectorRAGService()
ai = AIProvider(rag)

@router.put("/create_client")
def create_client(details: ClientListSetUp):
    return rag.initialise_client(details)