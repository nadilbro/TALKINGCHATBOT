from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict, Any
import base64
import traceback

from Providers.ai_provider import AIProvider
from Providers.voice_chat import VoiceChatSystem
from SQL.RAG import VectorRAGService

router = APIRouter(prefix="/voice", tags=["voice"])

rag = VectorRAGService()
ai = AIProvider(rag)
tts = VoiceChatSystem()


class VoiceInit(BaseModel):
    site_id: str


@router.post("/audio_chat_init")
async def audio_chat_init(details: VoiceInit):
    site_id = details.site_id

    # Uses your RAG.get_voice_init we added earlier
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
        payload = {}
        try:
            import json
            payload = json.loads(raw)
        except Exception:
            await ws.send_json({"type": "error", "message": "Invalid JSON payload"})
            await ws.close()
            return

        site_id = str(payload.get("site_id") or "").strip()
        user_text = str(payload.get("message") or "").strip()
        voice_name = str(payload.get("voice_name") or "").strip()

        past_messages = payload.get("pastMessages") or []
        past_answers = payload.get("pastAnswers") or []

        if not site_id or not user_text:
            await ws.send_json({"type": "error", "message": "Missing site_id or message"})
            await ws.close()
            return

        # 1) Generate bot text (your AIProvider should already do this)
        bot_text = await ai.get_chat_response(
            site_id=site_id,
            message=user_text,
            pastMessages=past_messages,
            pastAnswers=past_answers,
        )

        # Send text ASAP so UI updates quickly
        await ws.send_json({"type": "text", "text": bot_text})

        # 2) TTS -> MP3 bytes + visemes
        audio_bytes, visemes = await tts.synthesize_mp3_with_visemes(
            text=bot_text,
            voice_name=voice_name,
        )

        # 3) Send visemes (send early; your frontend buffers until audio "playing")
        for v in visemes:
            await ws.send_json({
                "type": "viseme",
                "t_ms": int(v.get("t_ms", 0)),
                "viseme_id": int(v.get("viseme_id", 0)),
            })

        # 4) Stream audio in chunks
        CHUNK = 32_000
        for i in range(0, len(audio_bytes), CHUNK):
            chunk = audio_bytes[i:i+CHUNK]
            await ws.send_json({
                "type": "audio",
                "data_b64": base64.b64encode(chunk).decode("utf-8")
            })

        await ws.send_json({"type": "done"})
        await ws.close()

    except WebSocketDisconnect:
        # user closed tab / widget
        return
    except Exception as e:
        # IMPORTANT: surface the error to the frontend AND to Render logs
        print("❌ WS error:", repr(e))
        traceback.print_exc()

        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

        try:
            await ws.close()
        except Exception:
            pass
