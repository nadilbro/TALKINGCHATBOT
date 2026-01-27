from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import asyncio
import time
from Providers.APIContracts import StatusResponse, ChatBotEdits, SiteID
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
def edit_widget(req: ChatBotEdits):
    return rag.edit_appearence(req)

@router.get("/widget_information")
def get_widget_information(site_id: str = Query(...)):
    # should return dict like:
    # { "chatbot_name": "...", "widget_color": "...", "widget_size": "...", "border_radius": "...", "greeting": "..." }
    return rag.get_appearence(site_id)
