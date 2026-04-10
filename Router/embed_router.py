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
from Providers.APIContracts import TextIngestionRequest
router = APIRouter(prefix="/embed", tags=["embed"])
rag = VectorRAGService()
_bearer = HTTPBearer()
from fastapi import HTTPException, BackgroundTasks
from pydantic import BaseModel
from Providers.web_scraper import scrape_and_ingest_website

class ScrapeWebsiteRequest(BaseModel):
    website_url: str
    replace_existing: bool = True
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
    business_description: Optional[str] = None
    assistant_name: Optional[str] = "Assistant"
    website_url: Optional[str] = None
    assistant_version: Optional[str] = "professional"
    inner_color: Optional[str] = "#FFFFFF"
    outer_color: Optional[str] = "#000000"
    icon_size: Optional[str] = "medium"
    font: Optional[str] = "system-ui"
    font_size: Optional[str] = "medium"
    welcome_message: Optional[str] = None

class UpdateApiKeyRequest(BaseModel):
    business_name: Optional[str] = None
    avatar_name: Optional[str] = None
    system_prompt: Optional[str] = None
    monthly_limit: Optional[int] = None
    is_active: Optional[bool] = None
    business_description: Optional[str] = None
    assistant_name: Optional[str] = None
    website_url: Optional[str] = None
    assistant_version: Optional[str] = None
    inner_color: Optional[str] = None
    outer_color: Optional[str] = None
    icon_size: Optional[str] = None
    font: Optional[str] = None
    font_size: Optional[str] = None
    welcome_message: Optional[str] = None


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
        business_description=req.business_description,
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
        business_description=req.business_description,
        website_url=req.website_url,
        last_scrape_at=req.last_scrape_at,
        assistant_name=req.assistant_name,
        assistant_version=req.assistant_version, 
        inner_color=req.inner_color,
        outer_color=req.outer_color, 
        icon_size=req.icon_size, 
        font=req.font,
        font_size=req.font_size,
        welcome_message=req.welcome_message,
    )
    return {"success": True}


# @router.delete("/keys/get_info/{key}")
# async def get_api_key_info(key: str, user=Depends(verify_token)):
#     user_id = user["uid"]
#     key_data = rag.getApiKey(key)
#     if not key_data or key_data.get("owner_user_id") != user_id:
#         raise HTTPException(status_code=404, detail="API key not found")
#     returnreturn {
#         "doc_id": doc_id,
#         "filename": filename,
#         "chunks": len(chunks),
#     }


@router.get("/keys/{key}")
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
    avatar_name = key_data.get("avatar_name", "Mia Sterling")
    avatar = rag.getAvatarByName(avatar_name)
    assistant_name = key_data.get("assistant_name") or "Assistant"

    return {
        # Identity
        "business_name": key_data.get("business_name"),
        "assistant_name": assistant_name,
        "assistant_version": key_data.get("assistant_version", "professional"),
        "avatar_name": avatar_name,
        "rive_url": avatar.get("url") if avatar else None,
        "voice_name": avatar.get("voice") if avatar else None,
        "welcome_message": key_data.get("welcome_message") or f"Hi, I'm {assistant_name}. How can I help you today?",

        # Widget styling
        "inner_color": key_data.get("inner_color", "#FFFFFF"),
        "outer_color": key_data.get("outer_color", "#000000"),
        "icon_size": key_data.get("icon_size", "medium"),
        "font": key_data.get("font", "system-ui"),
        "font_size": key_data.get("font_size", "medium"),
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

#Business Credit Managers
# ---------------------------------------------------------------------------
# Business credit management — read-only endpoints
# ---------------------------------------------------------------------------

@router.get("/credits")
async def get_business_credits(user=Depends(verify_token)):
    """Returns the business credits balance for the logged-in developer."""
    user_id = user["uid"]
    credits = rag.getBusinessCredits(user_id)
    return {"business_credits": credits}


@router.get("/credits/usage")
async def get_credits_usage_summary(user=Depends(verify_token)):
    """
    Returns total usage across all of this developer's API keys,
    plus current credit balance. Useful for the dashboard overview.
    """
    user_id = user["uid"]
    credits = rag.getBusinessCredits(user_id)
    keys = rag.listApiKeys(user_id)

    total_conversations = sum(k.get("conversations_used", 0) for k in keys)
    total_keys = len(keys)
    active_keys = sum(1 for k in keys if k.get("is_active"))

    return {
        "business_credits": credits,
        "total_keys": total_keys,
        "active_keys": active_keys,
        "total_conversations": total_conversations,
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
    

@router.post("/keys/{key}/text")
async def ingest_text(
        key: str,
        req: TextIngestionRequest,
        user=Depends(verify_token),
    ):
    """Business pastes text directly — no file upload needed."""
    owner_data = rag.getApiKey(key)
    if not owner_data or owner_data.get("owner_user_id") != user["uid"]:
        raise HTTPException(status_code=404, detail="API key not found")

    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    chunks = _chunk_text(req.content)
    doc_id = str(uuid.uuid4())

    for i, chunk in enumerate(chunks):
        embedding = await rag.embedText(chunk)
        rag.storeDocumentChunk(
            doc_id=doc_id,
            api_key=key,
            chunk_index=i,
            content=chunk,
            embedding=embedding,
            filename=req.title,
        )

    return {
        "doc_id": doc_id,
        "title": req.title,
        "chunks": len(chunks),
    }





@router.post("/keys/{key}/scrape")
async def scrape_website(
    key: str,
    req: ScrapeWebsiteRequest,
    background_tasks: BackgroundTasks,
    user=Depends(verify_token),
):
    """
    Triggers a website scrape for the given API key. The scrape runs in the
    background so the API response is immediate. The customer can check the
    status via /keys/{key}/documents to see the chunks being populated.
    """
    user_id = user["uid"]
    key_data = rag.getApiKey(key)
    if not key_data or key_data.get("owner_user_id") != user_id:
        raise HTTPException(status_code=404, detail="API key not found")
    
    if not req.website_url.strip():
        raise HTTPException(status_code=400, detail="website_url is required")
    
    # Kick off the scrape in the background
    background_tasks.add_task(
        scrape_and_ingest_website,
        api_key=key,
        website_url=req.website_url,
        rag=rag,
        replace_existing=req.replace_existing,
    )
    
    # Save the website URL on the api_key so we know where it was scraped from
    try:
        rag.updateApiKey(key=key, website_url=req.website_url)
    except Exception as e:
        print(f"==> Could not save website_url on api_key: {e}")
    
    return {
        "success": True,
        "message": "Website scrape started. Chunks will appear shortly.",
        "website_url": req.website_url,
    }


@router.get("/keys/{key}/scrape/status")
async def get_scrape_status(
    key: str,
    user=Depends(verify_token),
):
    """
    Returns the current scrape status for an API key — how many website chunks
    have been ingested and what the last scraped URL was.
    """
    user_id = user["uid"]
    key_data = rag.getApiKey(key)
    if not key_data or key_data.get("owner_user_id") != user_id:
        raise HTTPException(status_code=404, detail="API key not found")
    
    # Count the documents tagged as website scrapes
    docs = rag.listDocuments(key)
    website_docs = [d for d in docs if (d.get("filename") or "").startswith("website:")]
    
    return {
        "website_url": key_data.get("website_url"),
        "scraped_documents": len(website_docs),
        "last_scrape_at": key_data.get("last_scrape_at"),
    }