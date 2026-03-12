from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict, Any
import base64
import traceback

from Providers.ai_provider import AIProvider
from Providers.voice_chat import VoiceChatSystem
from SQL.RAG import VectorRAGService
from Providers.APIContracts import SessionInit

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import traceback
from typing import Any, Dict, List, Tuple, Optional

router = APIRouter(prefix="/system", tags=["chat"])

rag = VectorRAGService()
ai = AIProvider(rag)

# Lazy TTS so app doesn't crash at import/startup if env vars are missing
tts = None

def get_tts():
    global tts
    if tts is None:
        tts = VoiceChatSystem()
    return tts

#Initialise a chat session
@router.post("/chat_init")
async def chat_init(init_details: SessionInit):
    userID = init_details.userID
    chatID = init_details.chat_id
    # Uses your RAG.get_voice_init we added earlier
    avatar_key, voice_name, welcome_message, rive_url = rag.get_avatar(userID, chatID)

    if avatar_key is None and voice_name is None and welcome_message is None and rive_url is None:
        return {"error": "Session not found"}
    
    chat_history = rag.get_history(userID, chatID)
    return {
        "avatar_key": avatar_key,
        "voice_name": voice_name,
        "welcome_message": welcome_message,
        "rive_url": rive_url,
        "chat_history": chat_history,
    }


def _as_str(x: Any) -> str:
    return (str(x) if x is not None else "").strip()

def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

@router.websocket("/audio_chat_ws")
#Enables Streaming
async def audio_chat_ws(ws: WebSocket):
    print("HIT audio_chat_ws")
    await ws.accept()

    try:
        while True:
            # -------------------------
            # Receive a message (robust)
            # -------------------------
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                return

            payload: Dict[str, Any]
            if "json" in msg and msg["json"] is not None:
                payload = msg["json"]
            elif "text" in msg and msg["text"]:
                try:
                    import json
                    payload = json.loads(msg["text"])
                except Exception:
                    await ws.send_json({"type": "error", "message": "Invalid JSON payload"})
                    continue
            else:
                await ws.send_json({"type": "error", "message": "Expected JSON payload"})
                continue

            # Optional: allow client to close gracefully
            if _as_str(payload.get("type")).lower() == "close":
                await ws.send_json({"type": "done"})
                await ws.close()
                return

            # -------------------------
            # Parse fields
            # -------------------------
            user_id = _as_str(payload.get("user_id") or payload.get("site_id"))  # TEMP: support old key
            chat_id = _as_str(payload.get("chat_id"))
            user_text = _as_str(payload.get("message"))
            voice_name = _as_str(payload.get("voice_name"))

            if not user_id or not chat_id or not user_text:
                await ws.send_json({"type": "error", "message": "Missing user_id/chat_id/message"})
                continue

            # -------------------------
            # Load recent history from DB (source of truth)
            # -------------------------
            try:
                history = rag.get_recent_messages(user_id=user_id, chat_id=chat_id, limit=20)
                # history should be: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"Failed to load history: {str(e)}"})
                continue

            # -------------------------
            # Persist the user message
            # -------------------------
            try:
                rag.add_message(chat_id=chat_id, role="user", content=user_text)
                rag.update_last_message(chat_id=chat_id, last_message=user_text)
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"Failed to save user message: {str(e)}"})
                continue

            # -------------------------
            # Build pastMessages/pastAnswers for your existing AI code
            # -------------------------
            past_messages: List[str] = []
            past_answers: List[str] = []

            for m in history:
                role = (m.get("role") or "").lower()
                content = m.get("content") or ""
                if role == "user":
                    past_messages.append(content)
                elif role == "assistant":
                    past_answers.append(content)

            # -------------------------
            # Generate bot text
            # -------------------------
            try:
                bot_text = await ai.get_chat_response(
                    site_id=user_id,               # keep arg name if your AIProvider expects it
                    message=user_text,
                    pastMessages=past_messages,
                    pastAnswers=past_answers,
                )
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"AI failed: {str(e)}"})
                continue

            # Send text ASAP so UI updates quickly
            await ws.send_json({"type": "text", "text": bot_text})

            # Persist assistant message
            try:
                rag.add_message(chat_id=chat_id, role="assistant", content=bot_text)
                rag.update_last_message(chat_id=chat_id, last_message=bot_text)
            except Exception as e:
                # Not fatal to the user experience, but good to surface
                await ws.send_json({"type": "error", "message": f"Failed to save assistant message: {str(e)}"})

            # -------------------------
            # TTS (optional)
            # -------------------------
            try:
                tts_instance = get_tts()
            except Exception as e:
                # Text is still delivered; just no audio
                await ws.send_json({"type": "done"})
                continue

            try:
                audio_bytes, visemes = await tts_instance.synthesize_mp3_with_visemes(
                    text=bot_text,
                    voice_name=voice_name,
                )
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"TTS synthesis failed: {str(e)}"})
                await ws.send_json({"type": "done"})
                continue

            # Visemes
            for v in (visemes or []):
                await ws.send_json({
                    "type": "viseme",
                    "t_ms": _as_int(v.get("t_ms"), 0),
                    "viseme_id": _as_int(v.get("viseme_id"), 0),
                })

            # Stream audio as binary
            await ws.send_json({
                "type": "audio_begin",
                "format": "mp3",
                "sample_rate_hz": 16000,
                "channels": 1,
            })

            CHUNK = 32_000
            for i in range(0, len(audio_bytes), CHUNK):
                await ws.send_bytes(audio_bytes[i:i + CHUNK])

            await ws.send_json({"type": "audio_end"})
            await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        return
    except Exception as e:
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