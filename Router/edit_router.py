from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import asyncio
import time
from Providers.APIContracts import StatusResponse, ChatBotEdits, SiteID, AddDataRequest, DeleteEmbeddingRequest
from SQL.RAG import VectorRAGService
from Providers.ai_provider import AIProvider
import os
from fastapi import UploadFile, File, Form
import tempfile
import tempfile
from typing import Optional, List


router = APIRouter(prefix="/edit_traits", tags=["edit"])

rag = VectorRAGService()
ai = AIProvider(rag)

@router.put("/edit_chatbot")
def edit_traits(req: ChatBotEdits): 
    return rag.edit_traits(req)


@router.put("/widget_appearance")
def edit_widget(req: ChatBotEdits):
    return rag.edit_appearence(req)

@router.get("/get_widget_information")
def get_widget_information(site_id: str = Query(...)):
    return rag.get_appearence(SiteID(site_id=site_id))

@router.post("/add_data")
async def add_data(info: AddDataRequest):
    await rag.add_data(info)
    return {"status": "ok"}

@router.get("/get_data")
def get_data(site_id: str = Query(...)):
    return rag.get_embedding_data(site_id)

@router.post("/delete_data")
def delete_data(req: DeleteEmbeddingRequest):
    return rag.delete_data(siteID=req.site_id, chunk_id=req.chunk_index)

