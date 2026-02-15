import os
import asyncio
from typing import List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
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

# Lazy init so missing env vars won't crash import / health checks
_tts: Optional[VoiceChatSystem] = None


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


@router.websocket("/audio_chat_ws")
async def audio_chat_ws(ws: WebSocket):
    await ws.accept()

    try:
        payload = await ws.receive_json()
        details = VoiceRequest(**payload)

        # 1) Build prompt/context + description
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

        # 2) AI response (still non-streaming; next step later is token streaming)
        text = await ai.chat(
            site_id=details.site_id,
            system=system,
            user=details.message,
        )

        # Send the text first (easy to debug / display while audio starts)
        await ws.send_json({"type": "text", "text": text})

        # 3) Stream TTS audio + visemes over the same WS
        out_q: asyncio.Queue[dict] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Run Azure SDK in a thread, push messages into out_q
        tts = get_tts()
        tts_task = asyncio.create_task(
            run_in_threadpool(tts.synthesize_to_ws_queue, loop, out_q, text, details.voice_name)
        )

        # Forward messages to the client until done/error/disconnect
        while True:
            msg = await out_q.get()
            mtype = msg.get("type")

            await ws.send_json(msg)

            if mtype in ("done", "error"):
                break

        # Ensure the thread task is finished
        await tts_task

    except WebSocketDisconnect:
        # Client left; nothing else to do
        return
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
