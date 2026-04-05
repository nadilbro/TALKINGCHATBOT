import os
import uuid
import hashlib
import secrets
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security
from pydantic import BaseModel
from typing import Optional
from SQL.SQLManager import VectorRAGService
from Providers.firebase_auth import verify_token

router = APIRouter(prefix="/embed", tags=["embed"])
rag = VectorRAGService()
_bearer = HTTPBearer()


# ---------------------------------------------------------------------------
# Auth — API key verification for embedded widget requests
# ---------------------------------------------------------------------------
async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> dict:
    """
    Used by the embedded widget — verifies the business API key.
    Returns the api_key row if valid.
    """
    key = credentials.credentials
    key_data = rag.getApiKey(key)
    if not key_data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not key_data.get("is_active"):
        raise HTTPException(status_code=403, detail="API key is disabled")
    return key_data


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class CreateApiKeyRequest(BaseModel):
    business_name: str
    avatar_name: Optional[str] = "Mia Sterling"
    system_prompt: Optional[str] = None
    monthly_limit: Optional[int] = 500


class UpdateApiKeyRequest(BaseModel):
    business_name: Optional[str] = None
    avatar_name: Optional[str] = None
    system_prompt: Optional[str] = None
    monthly_limit: Optional[int] = None
    is_active: Optional[bool] = None


# ---------------------------------------------------------------------------
# Dashboard endpoints — protected by Firebase auth (business owner)
# ---------------------------------------------------------------------------

@router.post("/keys")
async def create_api_key(req: CreateApiKeyRequest, user=Depends(verify_token)):
    """Creates a new API key for a business. Called from the developer dashboard."""
    user_id = user["uid"]
    key = secrets.token_urlsafe(32)  # generates a secure random API key
    rag.createApiKey(
        key=key,
        owner_user_id=user_id,
        business_name=req.business_name,
        avatar_name=req.avatar_name,
        system_prompt=req.system_prompt,
        monthly_limit=req.monthly_limit,
    )
    return {
        "api_key": key,
        "business_name": req.business_name,
        "avatar_name": req.avatar_name,
        "monthly_limit": req.monthly_limit,
    }


@router.get("/keys")
async def list_api_keys(user=Depends(verify_token)):
    """Lists all API keys for the logged in business owner."""
    user_id = user["uid"]
    keys = rag.listApiKeys(user_id)
    return {"keys": keys}


@router.patch("/keys/{key}")
async def update_api_key(key: str, req: UpdateApiKeyRequest, user=Depends(verify_token)):
    """Updates an API key — change avatar, prompt, limit, or disable it."""
    user_id = user["uid"]
    key_data = rag.getApiKey(key)
    if not key_data or key_data.get("owner_user_id") != user_id:
        raise HTTPException(status_code=404, detail="API key not found")
    rag.updateApiKey(
        key=key,
        business_name=req.business_name,
        avatar_name=req.avatar_name,
        system_prompt=req.system_prompt,
        monthly_limit=req.monthly_limit,
        is_active=req.is_active,
    )
    return {"success": True}


@router.delete("/keys/{key}")
async def delete_api_key(key: str, user=Depends(verify_token)):
    """Deletes an API key."""
    user_id = user["uid"]
    key_data = rag.getApiKey(key)
    if not key_data or key_data.get("owner_user_id") != user_id:
        raise HTTPException(status_code=404, detail="API key not found")
    rag.deleteApiKey(key)
    return {"success": True}


@router.get("/keys/{key}/usage")
async def get_api_key_usage(key: str, user=Depends(verify_token)):
    """Returns usage stats for an API key."""
    user_id = user["uid"]
    key_data = rag.getApiKey(key)
    if not key_data or key_data.get("owner_user_id") != user_id:
        raise HTTPException(status_code=404, detail="API key not found")
    return {
        "conversations_used": key_data.get("conversations_used", 0),
        "monthly_limit": key_data.get("monthly_limit", 500),
        "is_active": key_data.get("is_active", True),
    }


# ---------------------------------------------------------------------------
# Document upload — business uploads their knowledge base
# ---------------------------------------------------------------------------

@router.post("/keys/{key}/documents")
async def upload_document(
    key: str,
    file: UploadFile = File(...),
    user=Depends(verify_token),
):
    """
    Uploads a document (PDF or text) to the business knowledge base.
    The document is chunked, embedded, and stored for RAG retrieval.
    """
    user_id = user["uid"]
    key_data = rag.getApiKey(key)
    if not key_data or key_data.get("owner_user_id") != user_id:
        raise HTTPException(status_code=404, detail="API key not found")

    content = await file.read()
    filename = file.filename or "document"

    # Extract text based on file type
    if filename.endswith(".pdf"):
        text = await _extract_pdf_text(content)
    else:
        text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from document")

    # Chunk and embed
    chunks = _chunk_text(text)
    doc_id = str(uuid.uuid4())

    for i, chunk in enumerate(chunks):
        embedding = await rag.embedText(chunk)
        rag.storeDocumentChunk(
            doc_id=doc_id,
            api_key=key,
            chunk_index=i,
            content=chunk,
            embedding=embedding,
            filename=filename,
        )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks": len(chunks),
    }


@router.get("/keys/{key}/documents")
async def list_documents(key: str, user=Depends(verify_token)):
    """Lists all documents uploaded for an API key."""
    user_id = user["uid"]
    key_data = rag.getApiKey(key)
    if not key_data or key_data.get("owner_user_id") != user_id:
        raise HTTPException(status_code=404, detail="API key not found")
    docs = rag.listDocuments(key)
    return {"documents": docs}


@router.delete("/keys/{key}/documents/{doc_id}")
async def delete_document(key: str, doc_id: str, user=Depends(verify_token)):
    """Deletes a document and all its chunks."""
    user_id = user["uid"]
    key_data = rag.getApiKey(key)
    if not key_data or key_data.get("owner_user_id") != user_id:
        raise HTTPException(status_code=404, detail="API key not found")
    rag.deleteDocument(doc_id)
    return {"success": True}


# ---------------------------------------------------------------------------
# Widget endpoints — called by the embedded iframe, authed by API key
# ---------------------------------------------------------------------------

@router.get("/config")
async def get_embed_config(key_data=Depends(verify_api_key)):
    """
    Called by the embed widget on load to get avatar config.
    Returns everything the widget needs to initialise.
    """
    avatar_name = key_data.get("avatar_name", "Mia Sterling")
    avatar = rag.getAvatarByName(avatar_name)

    return {
        "business_name": key_data.get("business_name"),
        "avatar_name": avatar_name,
        "rive_url": avatar.get("url") if avatar else None,
        "voice_name": avatar.get("voice") if avatar else None,
        "welcome_message": f"Hi, I'm {avatar_name}. How can I help you today?",
        "system_prompt": key_data.get("system_prompt") or (avatar.get("prompt") if avatar else ""),
    }


@router.post("/chat")
async def embed_chat(
    payload: dict,
    key_data=Depends(verify_api_key),
):
    """
    Text-only chat endpoint for the embedded widget.
    Returns text response with RAG context injected.
    For voice use the WebSocket endpoint.
    """
    api_key = key_data.get("key")
    monthly_limit = key_data.get("monthly_limit", 500)
    conversations_used = key_data.get("conversations_used", 0)

    if conversations_used >= monthly_limit:
        raise HTTPException(status_code=429, detail="Monthly conversation limit reached")

    user_message = payload.get("message", "")
    if not user_message:
        raise HTTPException(status_code=400, detail="Missing message")

    # RAG — retrieve relevant context from uploaded documents
    rag_context = ""
    try:
        embedding = await rag.embedText(user_message)
        chunks = rag.searchDocumentChunks(api_key=api_key, embedding=embedding, limit=3)
        if chunks:
            rag_context = "\n\n".join([c["content"] for c in chunks])
    except Exception as e:
        print(f"==> RAG retrieval failed: {e}")

    # Increment conversation count
    rag.incrementConversationCount(api_key)

    return {
        "rag_context": rag_context,  # frontend injects this into the system prompt
        "conversations_remaining": monthly_limit - conversations_used - 1,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits text into overlapping chunks for embedding."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


async def _extract_pdf_text(content: bytes) -> str:
    """Extracts text from PDF bytes."""
    try:
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        # fallback — treat as plain text
        return content.decode("utf-8", errors="ignore")