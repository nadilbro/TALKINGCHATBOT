from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
import time
from Providers.APIContracts import StatusResponse, WidgetAppearance, ChatBotEdits 
from SQL.RAG import VectorRAGService
from Providers.ai_provider import AIProvider
import os
from fastapi import UploadFile, File, Form
import tempfile
import tempfile
from typing import Optional 


router = APIRouter(prefix="/edit_traits", tags=["edit"])

rag = VectorRAGService()
ai = AIProvider(rag)

@router.put("/edit_chatbot")
def edit_traits(req: ChatBotEdits): 
    return rag.edit_traits(req)


@router.put("/widget_appearance")
def edit_widget(req: WidgetAppearance):
    return rag.edit_appearence(req)