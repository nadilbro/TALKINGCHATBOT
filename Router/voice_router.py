import re
import asyncio
from html import unescape
import traceback
from typing import Any, List
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from Providers.firebase_auth import verify_token
from fastapi import Depends
from Providers.ai_provider import AIProvider
from Providers.voice_chat import VoiceChatSystem
from SQL.SQLManager import VectorRAGService
from Providers.APIContracts import SessionInit
from Providers.firebase_auth import verify_ws_token
from Providers.web_search import TavilyProvider
from Providers.summary_generator import RollingSummaryManager
from Providers.STT import DeepgramProvider
from Providers.Account_Manager import AccountManager
router = APIRouter(prefix="/system", tags=["chat"])

summary_mgr = None
rag = VectorRAGService()
ai = AIProvider(rag)
account_manager = AccountManager(rag)
tts = None
tav = None
stt = None

MINUTES_PER_CREDIT = 7

def html_to_plain_text(html_text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", html_text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()

def get_tts():
    global tts
    if tts is None:
        tts = VoiceChatSystem()
    return tts

def get_stt():
    global stt
    if stt is None:
        stt = DeepgramProvider()
    return stt

def get_web_search():
    global tav
    if tav is None:
        tav = TavilyProvider()
    return tav

def get_summary_manager():
    global summary_mgr
    if summary_mgr is None:
        summary_mgr = RollingSummaryManager(
            gemini_provider=ai._providers["gemini_flash"],
            rag=rag
        )
    return summary_mgr

def _as_str(x: Any) -> str:
    return (str(x) if x is not None else "").strip()

def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


async def _run_tts(sentences: list, voice_id: str) -> tuple[bytes, list, float]:
    """
    Runs ElevenLabs TTS in parallel for all sentences.
    Returns (all_audio_bytes, all_visemes, total_duration_seconds).
    """
    tts_instance = get_tts()
    results = await asyncio.gather(
        *[tts_instance.synthesize_sentence(s, voice_id) for s in sentences],
        return_exceptions=True
    )

    all_audio = b""
    all_visemes = []
    cumulative_offset_ms = 0
    total_duration_seconds = 0.0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"==> Sentence {i} TTS failed: {result}", flush=True)
            continue
        audio_bytes_chunk, visemes, duration = result
        for v in visemes:
            all_visemes.append({
                "t_ms": v["t_ms"] + cumulative_offset_ms,
                "viseme_id": v["viseme_id"],
            })
        all_audio += audio_bytes_chunk
        cumulative_offset_ms += int(duration * 1000)
        total_duration_seconds += duration

    return all_audio, all_visemes, total_duration_seconds


async def _send_audio(ws: WebSocket, all_audio: bytes, all_visemes: list):
    """Sends audio_begin, audio chunks, and audio_end over the WebSocket."""
    await ws.send_json({
        "type": "audio_begin",
        "format": "mp3",
        "sample_rate_hz": 44100,
        "channels": 1,
        "visemes": [
            {
                "t_ms": _as_int(v.get("t_ms"), 0),
                "viseme_id": _as_int(v.get("viseme_id"), 0),
            }
            for v in all_visemes
        ],
    })
    CHUNK = 32_000
    for i in range(0, len(all_audio), CHUNK):
        await ws.send_bytes(all_audio[i:i + CHUNK])
    await ws.send_json({"type": "audio_end"})





# -----------------------------------------------------------------------
# CHAT INIT
# -----------------------------------------------------------------------

@router.post("/chat_init")
async def chat_init(init_details: SessionInit, user=Depends(verify_token)):
    userID = init_details.userID
    chatID = init_details.chat_id

    avatar_key = ""
    voice_name = ""
    welcome_message = ""
    rive_url = ""
    prompt = ""

    raw_history = rag.get_history(userID, chatID)

    result = rag.get_avatar(userID, chatID)
    if result:
        a_key, v_name, w_msg, r_url, r_prompt = result
        avatar_key = a_key or ""
        voice_name = v_name or ""
        welcome_message = w_msg or ""
        rive_url = r_url or ""
        prompt = r_prompt or ""

    chat_history = []
    for m in raw_history:
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if content:
            chat_history.append({"role": role, "content": content})

    return {
        "avatar_key": avatar_key,
        "voice_name": voice_name,
        "welcome_message": welcome_message,
        "rive_url": rive_url,
        "chat_history": chat_history,
        "prompt": prompt
    }


# -----------------------------------------------------------------------
# MAIN CHAT WEBSOCKET
# -----------------------------------------------------------------------

@router.websocket("/audio_chat_ws")
async def audio_chat_ws(ws: WebSocket):
    print("HIT audio_chat_ws")
    await ws.accept()
    try:
        user = await verify_ws_token(ws)
    except ValueError:
        return

    try:
        while True:
            try:
                payload = await ws.receive_json()
            except WebSocketDisconnect:
                return
            except Exception:
                await ws.send_json({"type": "error", "message": "Invalid JSON payload"})
                continue

            if _as_str(payload.get("type")).lower() == "close":
                await ws.send_json({"type": "done"})
                await ws.close()
                return

            user_id = _as_str(payload.get("user_id") or payload.get("site_id"))
            chat_id = _as_str(payload.get("chat_id"))
            user_text = _as_str(payload.get("message"))
            voice_id = _as_str(payload.get("voice_name"))
            prompt = _as_str(payload.get("prompt"))
            web_search = _as_str(payload.get("web_search"))
            audio_on = payload.get("voice_on", True)
            raw_audio = payload.get("audio_bytes")

            if raw_audio and "," in raw_audio:
                raw_audio = raw_audio.split(",", 1)[1]
            audio_bytes = base64.b64decode(raw_audio) if raw_audio else None

            # CHECK CREDITS
            if not rag.hasEnoughCredits(user_id):
                await ws.send_json({
                    "type": "error",
                    "message": "You have no credits remaining. Please top up to continue chatting.",
                    "code": "NO_CREDITS"
                })
                await ws.send_json({"type": "done"})
                continue

            # STT
            if audio_bytes:
                print(f"==> Audio bytes length: {len(audio_bytes)}")
                try:
                    stt_instance = get_stt()
                    user_text = stt_instance.get_transcript(audio_bytes)
                    print(f"==> Transcript result: '{user_text}'")
                    if not user_text:
                        await ws.send_json({"type": "error", "message": "Could not understand audio. Try again."})
                        continue
                    await ws.send_json({"type": "transcript", "text": user_text})
                except Exception as e:
                    print(f"==> STT exception: {e}")
                    traceback.print_exc()
                    await ws.send_json({"type": "error", "message": f"Transcription failed: {e}"})
                    continue

            if not user_id or not chat_id or (not user_text and not audio_bytes):
                await ws.send_json({"type": "error", "message": "Missing user_id/chat_id/message"})
                continue

            if not voice_id:
                try:
                    _, voice_id, _, _, _ = rag.get_avatar(user_id, chat_id)
                    voice_id = _as_str(voice_id)
                except Exception:
                    voice_id = "UgBBYS2sOqTuMpoF3BR0"

            try:
                history = rag.get_recent_messages(user_id=user_id, chat_id=chat_id, limit=20)
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"Failed to load history: {str(e)}"})
                continue

            try:
                rag.add_message(chat_id=chat_id, role="user", content=user_text)
                rag.update_last_message(chat_id=chat_id, last_message=user_text)
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"Failed to save user message: {str(e)}"})
                continue

            # Build prompt
            smgr = get_summary_manager()
            summary_context = smgr.build_context(chat_id)
            system_prompt = f"{prompt}\n\n{summary_context}" if summary_context else prompt

            recent_history = history[-3:] if len(history) > 3 else history

            if web_search:
                tavily_instance = get_web_search()
                web_response = tavily_instance.web_search(user_text, 3)
                system_prompt = f"{system_prompt}\n\n{web_response}"

            history_lines = []
            for m in recent_history:
                role = (m.get("role") or "").lower()
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if role == "user":
                    history_lines.append(f"User: {content}")
                elif role == "assistant":
                    history_lines.append(f"Assistant: {content}")
                else:
                    history_lines.append(f"{role.title()}: {content}")

            conversation_history = "\n".join(history_lines)
            if conversation_history or summary_context:
                user_prompt = f"Conversation history:\n{conversation_history}\n\nLatest user message:\n{user_text}"
            else:
                user_prompt = user_text

              # ----------------------------------------------------------
            # Check if diagrams are enabled for this user
            # ----------------------------------------------------------
            try:
                diagrams_enabled = rag.get_diagram_usage(user_id)
            except Exception:
                diagrams_enabled = False

            # ----------------------------------------------------------
            # Generate chat response first (needed as context for diagram)
            # ----------------------------------------------------------
            async def _generate_chat():
                sentences = []
                sentence_buffer = ""
                async for delta in ai.stream(site_id=user_id, system=system_prompt, user=user_prompt):
                    sentence_buffer += delta
                    while re.search(r'[.?!,]\s', sentence_buffer):
                        match = re.search(r'[.?!,]\s', sentence_buffer)
                        cut = match.end()
                        sentence = sentence_buffer[:cut].strip()
                        sentence_buffer = sentence_buffer[cut:]
                        if sentence and len(sentence) > 2:
                            sentences.append(sentence)
                if sentence_buffer.strip() and len(sentence_buffer.strip()) > 2:
                    sentences.append(sentence_buffer.strip())
                bot_text = " ".join(sentences)
                return sentences, bot_text

            try:
                sentences, bot_text = await _generate_chat()
                if not sentences:
                    await ws.send_json({"type": "error", "message": "No response generated"})
                    await ws.send_json({"type": "done"})
                    continue
                print(f"==> {len(sentences)} sentences", flush=True)
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"AI failed: {str(e)}"})
                continue

            # Send the spoken text immediately
            await ws.send_json({"type": "text", "text": bot_text})

            # Save assistant message and update rolling summary
            try:
                rag.add_message(chat_id=chat_id, role="assistant", content=bot_text)
                rag.update_last_message(chat_id=chat_id, last_message=bot_text)
            except Exception:
                pass

            try:
                smgr = get_summary_manager()
                recent_for_summary = history[-6:] if len(history) > 6 else list(history)
                recent_for_summary.append({"role": "user", "content": user_text})
                recent_for_summary.append({"role": "assistant", "content": bot_text})
                await smgr.on_new_message(chat_id, recent_for_summary[-6:])
            except Exception as e:
                print(f"Summary update error: {e}")

            # ----------------------------------------------------------
            # Define the diagram task — uses bot_text as context so the
            # diagram illustrates exactly what the avatar is saying
            # ----------------------------------------------------------
            async def _generate_diagram_with_context():
                if not diagrams_enabled:
                    print("Diagrams disabled for this user")
                    return None
                try:
                    print("Generating diagram...")
                    diagram_input = (
                        f"User's question: {user_text}\n\n"
                        f"Spoken answer that was just given to the user:\n{bot_text}\n\n"
                        f"Draw a diagram that illustrates the spoken answer above. "
                        f"The diagram must match the structure, steps, and terminology "
                        f"of the spoken answer. Do not invent new steps or use different "
                        f"labels. If a diagram does not meaningfully illustrate this "
                        f"answer, output NONE."
                    )
                    raw = await ai.get_diagram(site_id=user_id, user=diagram_input)
                    print(f"==> Raw diagram response: {raw[:500]!r}")
                    if not raw:
                        return None
                    if raw.strip().upper().startswith("NONE"):
                        print("Diagram model returned NONE")
                        return None
                    match = re.search(r'<svg.*?</svg>', raw, re.DOTALL | re.IGNORECASE)
                    if not match:
                        print("No <svg> block found in diagram response")
                        return None
                    return match.group(0)
                except Exception as e:
                    print(f"==> Diagram generation failed: {e}", flush=True)
                    return None

            # ----------------------------------------------------------
            # Kick off diagram in the background — it runs while TTS runs
            # ----------------------------------------------------------
            diagram_task = asyncio.create_task(_generate_diagram_with_context())

            # ----------------------------------------------------------
            # TTS — only if audio is on
            # ----------------------------------------------------------
            if audio_on:
                try:
                    print(f"==> Firing {len(sentences)} ElevenLabs tasks in parallel", flush=True)
                    all_audio, all_visemes, total_duration_seconds = await _run_tts(sentences, voice_id)

                    if not all_audio:
                        await ws.send_json({"type": "error", "message": "TTS produced no audio"})
                    else:
                        await _send_audio(ws, all_audio, all_visemes)

                        # Cost tracking
                        try:
                            cost = account_manager.processUsedCost(
                                outputText=bot_text,
                                inputText=user_prompt,
                                SST_Length_seconds=len(audio_bytes) / 16000 if audio_bytes else 0,
                                webSearch=bool(web_search),
                                voice_on=bool(audio_on),
                            )
                            credits_used = cost / 0.15
                            remaining = rag.deductCredits(user_id, credits_used)
                            print(f"==> Cost: ${cost:.4f} | Deducted {credits_used:.4f} credits. Remaining: {remaining}", flush=True)
                        except Exception as e:
                            print(f"==> Cost tracking failed: {e}", flush=True)

                except Exception as e:
                    print(f"==> TTS error: {e}", flush=True)
                    traceback.print_exc()
                    await ws.send_json({"type": "error", "message": f"TTS failed: {str(e)}"})

            # ----------------------------------------------------------
            # Now await the diagram — likely already done by this point
            # ----------------------------------------------------------
            try:
                svg = await diagram_task
            except Exception as e:
                print(f"==> Diagram task error: {e}", flush=True)
                svg = None

            if svg:
                await ws.send_json({"type": "diagram", "svg": svg})

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


# -----------------------------------------------------------------------
# EMBED CHAT WEBSOCKET
# -----------------------------------------------------------------------

@router.websocket("/embed_chat_ws")
async def embed_chat_ws(ws: WebSocket):
    """
    WebSocket endpoint for embedded widget.
    Auth via API key in query param. No credit checks — billing per API key.
    """
    print("HIT embed_chat_ws")
    await ws.accept()

    api_key = ws.query_params.get("api_key")
    if not api_key:
        await ws.close(code=4001, reason="Missing api_key")
        return

    # Initial validation only — full key_data is re-fetched per message
    initial_key_data = rag.getApiKey(api_key)
    if not initial_key_data or not initial_key_data.get("is_active"):
        await ws.close(code=4001, reason="Invalid or inactive API key")
        return

    FORMATTING_RULE = (
        "CRITICAL FORMATTING RULE: Never use markdown formatting of any kind. "
        "No asterisks, no bold, no headers, no hashtags, no bullet points, no numbered lists, "
        "no dashes, no colons, no semicolons. Write in plain conversational paragraphs only. "
        "This is spoken aloud, not read on screen."
    )

    print(f"==> embed WS opened for api_key={api_key}")

    try:
        while True:
            try:
                payload = await ws.receive_json()
            except WebSocketDisconnect:
                return
            except Exception:
                await ws.send_json({"type": "error", "message": "Invalid JSON payload"})
                continue

            if _as_str(payload.get("type")).lower() == "close":
                await ws.send_json({"type": "done"})
                await ws.close()
                return

            # RE-FETCH key_data on every message so dashboard toggles take effect mid-session
            key_data = rag.getApiKey(api_key)
            if not key_data or not key_data.get("is_active"):
                await ws.send_json({"type": "error", "message": "API key deactivated", "code": "INVALID_KEY"})
                await ws.close()
                return

            if key_data.get("conversations_used", 0) >= key_data.get("monthly_limit", 500):
                await ws.send_json({"type": "error", "message": "Monthly conversation limit reached", "code": "LIMIT_REACHED"})
                await ws.close()
                return

            business_name = key_data.get("business_name") or ""
            business_description = key_data.get("business_description") or ""
            avatar_name = key_data.get("avatar_name") or "Mia Sterling"

            # Safer default: if column is NULL or missing, treat as OFF
            personality_on = bool(key_data.get("personality_on"))

            user_text = _as_str(payload.get("message"))
            voice_id = _as_str(payload.get("voice_name"))
            audio_on = payload.get("voice_on", True)
            raw_audio = payload.get("audio_bytes")

            if raw_audio and "," in raw_audio:
                raw_audio = raw_audio.split(",", 1)[1]
            audio_bytes = base64.b64decode(raw_audio) if raw_audio else None

            # STT
            if audio_bytes:
                try:
                    stt_instance = get_stt()
                    user_text = stt_instance.get_transcript(audio_bytes)
                    if not user_text:
                        await ws.send_json({"type": "error", "message": "Could not understand audio."})
                        continue
                    await ws.send_json({"type": "transcript", "text": user_text})
                except Exception as e:
                    await ws.send_json({"type": "error", "message": f"Transcription failed: {e}"})
                    continue

            if not user_text:
                await ws.send_json({"type": "error", "message": "Missing message"})
                continue

            # Look up avatar once — used for voice and (maybe) prompt
            avatar = rag.getAvatarByName(avatar_name)

            # Resolve voice
            if not voice_id:
                voice_id = (avatar.get("voice") if avatar else None) or "UgBBYS2sOqTuMpoF3BR0"

            # RAG lookup
            rag_context = ""
            try:
                embedding = await rag.embedText(user_text)
                chunks = rag.searchDocumentChunks(api_key=api_key, embedding=embedding, limit=3)
                if chunks and chunks[0].get("similarity", 0) >= 0.3:
                    rag_context = "\n".join(f"- {c['content']}" for c in chunks)
                print(f"==> RAG similarity: {chunks[0].get('similarity', 0) if chunks else 0:.3f}")
            except Exception as e:
                print(f"==> RAG failed: {e}")

            # Build system prompt — explicit branches, no fallback bleed
            if personality_on:
                avatar_prompt = (avatar.get("prompt") if avatar else "") or ""
                fallback_prompt = key_data.get("system_prompt") or ""
                system_prompt = avatar_prompt or fallback_prompt
            else:
                system_prompt = (
                    f"{FORMATTING_RULE}\n\n"
                    f"You are a helpful support agent for {business_name}. "
                    f"{business_description} "
                    f"Answer the user's questions accurately and helpfully. "
                    f"Do not make up information you do not have. "
                    f"If you don't know the answer, say so honestly."
                )

            if rag_context:
                system_prompt = f"{system_prompt}\n\nRelevant information from our knowledge base:\n{rag_context}"

            # DEBUG — remove once confirmed working
            print(f"==> personality_on={personality_on} | prompt_preview={system_prompt[:200]!r}")

            # Stream Gemini
            try:
                sentences = []
                sentence_buffer = ""
                async for delta in ai.stream(site_id=api_key, system=system_prompt, user=user_text):
                    sentence_buffer += delta
                    while re.search(r'[.?!,]\s', sentence_buffer):
                        match = re.search(r'[.?!,]\s', sentence_buffer)
                        cut = match.end()
                        sentence = sentence_buffer[:cut].strip()
                        sentence_buffer = sentence_buffer[cut:]
                        if sentence and len(sentence) > 2:
                            sentences.append(sentence)
                if sentence_buffer.strip() and len(sentence_buffer.strip()) > 2:
                    sentences.append(sentence_buffer.strip())

                if not sentences:
                    await ws.send_json({"type": "error", "message": "No response generated"})
                    await ws.send_json({"type": "done"})
                    continue

                bot_text = " ".join(sentences)
                print(f"==> embed {len(sentences)} sentences", flush=True)

            except Exception as e:
                await ws.send_json({"type": "error", "message": f"AI failed: {str(e)}"})
                continue

            await ws.send_json({"type": "text", "text": bot_text})

            try:
                rag.incrementConversationCount(api_key)
            except Exception as e:
                print(f"==> Failed to increment conversation count: {e}")

            # TTS
            if audio_on:
                try:
                    print(f"==> Embed firing {len(sentences)} ElevenLabs tasks in parallel", flush=True)
                    all_audio, all_visemes, _ = await _run_tts(sentences, voice_id)

                    if not all_audio:
                        await ws.send_json({"type": "error", "message": "TTS produced no audio"})
                        await ws.send_json({"type": "done"})
                        continue

                    await _send_audio(ws, all_audio, all_visemes)

                except Exception as e:
                    print(f"==> Embed TTS error: {e}")
                    traceback.print_exc()
                    await ws.send_json({"type": "error", "message": f"TTS failed: {str(e)}"})

            await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        return
    except Exception as e:
        print("❌ Embed WS error:", repr(e))
        traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass