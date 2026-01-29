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
import traceback
from fastapi import HTTPException

router = APIRouter(prefix="/edit_traits", tags=["edit"])

rag = VectorRAGService()
ai = AIProvider(rag)

@router.put("/edit_chatbot")
def edit_traits(req: ChatBotEdits): 
    return rag.edit_traits(req)


@router.put("/widget_appearance")
def edit_widget(req: ChatBotEdits):
    return rag.edit_appearence(req)


@router.post("/add_data")
async def add_data(info: AddDataRequest):
    print("🔥 /edit_traits/add_data HIT", info.site_id, info.source, len(info.text or ""))
    try:
        await rag.add_data(info)
        return {"status": "ok"}
    except Exception as e:
        print("❌ add_data failed:", repr(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get_data")
#This is to get the embedding data
def get_data(site_id: str = Query(...)):
    return rag.get_embedding_data(site_id)

@router.post("/delete_data")
def delete_data(req: DeleteEmbeddingRequest):
    return rag.delete_data(siteID=req.site_id, chunk_id=req.chunk_index)

