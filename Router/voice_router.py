import os
import asyncio
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from Providers.APIContracts import VoiceChat
from Providers.ai_provider import AIProvider
from Providers.voice_chat import VoiceChatSystem
from Security.signing import Security
from SQL.RAG import VectorRAGService


router = APIRouter(prefix="/voice", tags=["voice"])

rag = VectorRAGService()
ai = AIProvider(rag)
security = Security()

# Lazy-init so missing env vars don't crash app import/startup
_tts: VoiceChatSystem | None = None


def get_tts() -> VoiceChatSystem:
    global _tts
    if _tts is None:
        _tts = VoiceChatSystem()
    return _tts


class VoiceInit(BaseModel):
    site_id: str


class VoiceRequest(BaseModel):
    site_id: str
    voice_name: str
    message: str
    pastMessages: List[str] = Field(default_factory=list)
    pastAnswers: List[str] = Field(default_factory=list)


@router.post("/audio_chat_init")
async def audio_chat_init(details: VoiceInit):
    avatar_key, voice_name, welcome_message, primary_color = rag.get_avatar(details.site_id)
    if not avatar_key:
        raise HTTPException(status_code=404, detail="No avatar configured for this site_id")

    worker_base = os.getenv("AVATAR_WORKER_BASE")
    secret = os.getenv("AVATAR_SIGNING_SECRET")
    if not worker_base or not secret:
        raise HTTPException(status_code=500, detail="Avatar signing env vars not set")

    rive_url = security.sign_avatar(worker_base, avatar_key, secret, ttl=300)
    return VoiceChat(
        site_id=details.site_id,
        rive_url=rive_url,
        voice_name=voice_name,
        primary_color=primary_color,
        welcome_message=welcome_message,
    )


def _build_system_prompt(description: str, context: str, past_q: List[str], past_a: List[str]) -> str:
    return f"""
You are an AI assistant for the following business:
{description}

Write responses that are meant to be spoken out loud.
Use natural, conversational language.
Keep sentences short and easy to understand when heard.

Rules:
- Do not use markdown, emojis, HTML, or lists.
- Do not mention being an AI or reference prompts or context.
- Answer clearly and directly.
- Be concise, but include all necessary information.
- Be polite and do not use any inappropriate language.
- If something is uncertain, say so briefly and naturally.

Relevant context:
{context}

Conversation history (most recent last):
Questions: {past_q}
Answers: {past_a}
""".strip()


@router.post("/audio_chat_stream")
async def audio_chat_stream(details: VoiceRequest):
    """
    Returns a live MP3 audio stream (audio/mpeg).
    Perceived latency improves because audio starts flowing as soon as Azure produces bytes.
    """
    # 1) Build prompt/context + business description
    try:
        (prompt, _), description = await asyncio.gather(
            rag.process_question(details.message, details.site_id, 2),
            run_in_threadpool(rag.get_client, "description", details.site_id),
        )

        system = _build_system_prompt(
            description=description,
            context=prompt.context,
            past_q=details.pastMessages,
            past_a=details.pastAnswers,
        )

        # 2) AI response (still non-streaming)
        text = await ai.chat(
            site_id=details.site_id,
            system=system,
            user=details.message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {e}")

    # 3) Stream TTS audio
    def audio_gen():
        # generator yields bytes
        for chunk in get_tts().synthesize_stream_mp3(text, details.voice_name):
            yield chunk

    # You can optionally expose the text via header for debugging (keep it small)
    headers = {"X-Response-Text": text[:400]}

    return StreamingResponse(
        audio_gen(),
        media_type="audio/mpeg",
        headers=headers
    )
