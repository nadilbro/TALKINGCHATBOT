import os
import json
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from Providers.voice_chat import VoiceChatSystem
from SQL.RAG import VectorRAGService
from Providers.ai_provider import AIProvider


router = APIRouter(prefix="/voice", tags=["voice"])

rag = VectorRAGService()
ai = AIProvider(rag)
tts = VoiceChatSystem()


class VoiceInit(BaseModel):
    site_id: str


class VoiceWSRequest(BaseModel):
    site_id: str
    message: str
    voice_name: str = ""
    pastMessages: List[str] = []
    pastAnswers: List[str] = []


@router.post("/audio_chat_init")
async def audio_chat_init(details: VoiceInit):
    """
    Returns voice settings + optional Rive URL + welcome message.
    Adjust to your DB/RAG structure.
    """
    site_id = details.site_id

    # Example: your rag likely returns these fields
    # avatar_key, voice_name, welcome_message, primary_color = rag.get_voice_init(site_id)
    # Replace the below with your real lookup:
    avatar_key, voice_name, welcome_message, primary_color, rive_url = rag.get_voice_init(site_id)

    return {
        "avatar_key": avatar_key,
        "voice_name": voice_name,
        "welcome_message": welcome_message,
        "primary_color": primary_color,
        "rive_url": rive_url,
    }


@router.websocket("/audio_chat_ws")
async def audio_chat_ws(ws: WebSocket):
    await ws.accept()

    try:
        raw = await ws.receive_text()
        payload = VoiceWSRequest(**json.loads(raw))

        site_id = payload.site_id
        user_text = (payload.message or "").strip()
        voice_name = (payload.voice_name or "").strip()

        if not user_text:
            await ws.send_text(json.dumps({"type": "error", "message": "Empty message"}))
            await ws.close()
            return

        # 1) Get AI text response (adjust to your provider signature)
        #    Make sure this returns a plain string.
        bot_text = await ai.chat(
            site_id=site_id,
            message=user_text,
            past_messages=payload.pastMessages,
            past_answers=payload.pastAnswers,
        )

        if not isinstance(bot_text, str):
            bot_text = str(bot_text or "")

        # Send text first so UI can render immediately
        await ws.send_text(json.dumps({"type": "text", "text": bot_text}))

        # 2) Synthesize MP3 + visemes in a threadpool (blocking SDK)
        def _tts_blocking():
            return tts.synthesize_mp3_with_visemes(
                text=bot_text,
                voice_name=voice_name,
                timeout_ms=20000,
            )

        try:
            audio_bytes, visemes = await run_in_threadpool(_tts_blocking)
        except Exception as e:
            await ws.send_text(json.dumps({"type": "error", "message": f"TTS error: {e}"}))
            await ws.close()
            return

        # 3) Stream visemes (client buffers them until audio starts)
        for v in visemes:
            await ws.send_text(json.dumps({
                "type": "viseme",
                "t_ms": int(v.t_ms),
                "viseme_id": int(v.viseme_id),
            }))

        # 4) Stream MP3 chunks
        for chunk in tts.chunk_bytes(audio_bytes, chunk_size=8192):
            await ws.send_text(json.dumps({
                "type": "audio",
                "data_b64": tts.bytes_to_b64(chunk),
            }))

        await ws.send_text(json.dumps({"type": "done"}))
        await ws.close()

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            await ws.close()
        except Exception:
            pass
