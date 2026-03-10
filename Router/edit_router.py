from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import asyncio
import time
from Providers.APIContracts import ChatBotEdits
from SQL.RAG import VectorRAGService
from Providers.ai_provider import AIProvider
import os
from fastapi import UploadFile, File, Form
import tempfile
import tempfile
from typing import Optional, List
import traceback
from fastapi import HTTPException

router = APIRouter(prefix="/edit_traits", tags=["edit"])

rag = VectorRAGService()
ai = AIProvider(rag)

@router.put("/edit_chatbot")
def edit_settings(req: ChatBotEdits): 
    return rag.edit_traits(req)

