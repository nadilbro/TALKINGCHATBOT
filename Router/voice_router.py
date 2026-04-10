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
from Providers.APIContracts import SessionInit, DiagramInit
from Providers.firebase_auth import verify_ws_token
from Providers.web_search import TavilyProvider
from Providers.summary_generator import RollingSummaryManager
from Providers.STT import DeepgramProvider
import time
from Providers.Account_Manager import AccountManager
router = APIRouter(prefix="/system", tags=["chat"])
from Providers.file_extractor import FileExtractor
summary_mgr = None
rag = VectorRAGService()
ai = AIProvider(rag)
account_manager = AccountManager(rag)
fileE = FileExtractor(ai)
tts = None
tav = None
stt = None
MINUTES_PER_CREDIT = 7

def strip_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'```[\w]*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Clean up extra whitespace left by stripping
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def fix_markdown_formatting(text):
    # Add newlines around headers if they're inline
    text = re.sub(r'([^\n])(#{1,3}\s+)', r'\1\n\n\2', text)  # Before header
    text = re.sub(r'(#{1,3}\s+[^\n]+)([^\n])', r'\1\n\n\2', text)  # After header
    return text

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

            # ----------------------------------------------------------
            # PARSE PAYLOAD
            # ----------------------------------------------------------
            user_id    = _as_str(payload.get("user_id") or payload.get("site_id"))
            chat_id    = _as_str(payload.get("chat_id"))
            user_text  = _as_str(payload.get("message"))
            voice_id   = _as_str(payload.get("voice_name"))
            prompt     = _as_str(payload.get("prompt"))
            web_search = _as_str(payload.get("web_search"))
            audio_on   = payload.get("voice_on", True)
            pro_mode   = payload.get("pro_mode", False)
            raw_audio  = payload.get("audio_bytes")

            # Multi-file support with legacy single-file fallback
            files_payload = payload.get("files") or []
            if not files_payload:
                legacy_raw  = payload.get("file_bytes")
                legacy_name = payload.get("file_name")
                if legacy_raw and legacy_name:
                    files_payload = [{"file_bytes": legacy_raw, "file_name": legacy_name}]

            MAX_FILES             = 5
            MAX_TOTAL_UPLOAD_BYTES = 40 * 1024 * 1024  # 40MB

            if raw_audio and "," in raw_audio:
                raw_audio = raw_audio.split(",", 1)[1]
            audio_bytes = base64.b64decode(raw_audio) if raw_audio else None

            # ----------------------------------------------------------
            # CREDIT CHECK
            # ----------------------------------------------------------
            if not rag.hasEnoughCredits(user_id):
                await ws.send_json({"type": "error", "message": "You have no credits remaining. Please top up to continue chatting.", "code": "NO_CREDITS"})
                await ws.send_json({"type": "done"})
                continue

            # ----------------------------------------------------------
            # FILE COUNT CAP
            # ----------------------------------------------------------
            if len(files_payload) > MAX_FILES:
                await ws.send_json({"type": "error", "message": f"Max {MAX_FILES} files per message.", "code": "TOO_MANY_FILES"})
                await ws.send_json({"type": "done"})
                continue

            # ----------------------------------------------------------
            # SPEECH TO TEXT
            # ----------------------------------------------------------
            if audio_bytes:
                try:
                    stt_instance = get_stt()
                    user_text = stt_instance.get_transcript(audio_bytes)
                    if not user_text:
                        await ws.send_json({"type": "error", "message": "Could not understand audio. Try again."})
                        continue
                    await ws.send_json({"type": "transcript", "text": user_text})
                except Exception as e:
                    traceback.print_exc()
                    await ws.send_json({"type": "error", "message": f"Transcription failed: {e}"})
                    continue

            if not user_id or not chat_id or (not user_text and not audio_bytes):
                await ws.send_json({"type": "error", "message": "Missing user_id/chat_id/message"})
                continue

            # ----------------------------------------------------------
            # VOICE ID FALLBACK
            # ----------------------------------------------------------
            if not voice_id:
                try:
                    avatar_data = rag.get_avatar(user_id, chat_id)
                    if avatar_data:
                        voice_id = _as_str(avatar_data[1])
                except Exception:
                    pass
                if not voice_id:
                    voice_id = "UgBBYS2sOqTuMpoF3BR0"

            # ----------------------------------------------------------
            # LOAD HISTORY
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # PROCESS FILES
            # ----------------------------------------------------------
            file_context     = ""
            image_attachments = []
            file_text_parts  = []
            total_upload_bytes = 0
            upload_too_large = False

            for idx, f in enumerate(files_payload):
                raw_file  = f.get("file_bytes")
                file_name = f.get("file_name")

                if not raw_file or not file_name:
                    continue

                if "," in raw_file:
                    raw_file = raw_file.split(",", 1)[1]

                try:
                    file_bytes_decoded = base64.b64decode(raw_file)
                except Exception as e:
                    await ws.send_json({"type": "error", "message": f"Could not decode {file_name}."})
                    continue

                total_upload_bytes += len(file_bytes_decoded)
                if total_upload_bytes > MAX_TOTAL_UPLOAD_BYTES:
                    await ws.send_json({"type": "error", "message": f"Total upload exceeds {MAX_TOTAL_UPLOAD_BYTES // (1024*1024)}MB limit.", "code": "UPLOAD_TOO_LARGE"})
                    upload_too_large = True
                    break

                try:
                    ext = file_name.lower().split(".")[-1]
                    if ext in ("png", "jpg", "jpeg", "webp"):
                        # Image — pass directly to vision
                        mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
                        image_attachments.append({"mime_type": mime_map[ext], "data": file_bytes_decoded})
                        file_text_parts.append(f"[Image {idx + 1}: {file_name}]")
                        print(f"==> Image attached: {file_name}")
                    else:
                        # Document — extract + summarise
                        extracted = await fileE.extract_text(file_bytes_decoded, file_name)
                        if extracted and len(extracted) > 500:
                            summary_prompt = (
                                "Summarise the key information, questions, and any given answers from the following content "
                                "in a concise way that preserves all important values, equations, and steps. "
                                "Do not explain or elaborate, just extract and compress:\n\n"
                                + extracted
                            )
                            extracted = await ai.chat(site_id=user_id, system="You are a precise summariser.", user=summary_prompt)
                            print(f"==> File summarised: {file_name}, length={len(extracted)}")
                        file_text_parts.append(f"=== File {idx + 1}: {file_name} ===\n{extracted}")
                        print(f"==> File processed: {file_name}")
                except Exception as e:
                    traceback.print_exc()
                    await ws.send_json({"type": "error", "message": f"Could not read {file_name}: {e}"})

            if upload_too_large:
                await ws.send_json({"type": "done"})
                continue

            if file_text_parts:
                file_context = "\n\n".join(file_text_parts)
                print(f"==> Total file_context length: {len(file_context)}")

            # ----------------------------------------------------------
            # BUILD BASE PROMPT
            # ----------------------------------------------------------
            smgr = get_summary_manager()
            summary_context = smgr.build_context(chat_id)
            system_prompt = f"{prompt}\n\n{summary_context}" if summary_context else prompt

            recent_history = history[-3:] if len(history) > 3 else history

            if web_search:
                web_response = get_web_search().web_search(user_text, 3)
                system_prompt = f"{system_prompt}\n\n{web_response}"

            history_lines = []
            for m in recent_history:
                role    = (m.get("role") or "").lower()
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if role == "user":
                    history_lines.append(f"User: {content}")
                elif role == "assistant":
                    history_lines.append(f"Assistant: {content}")

            conversation_history = "\n".join(history_lines)
            if conversation_history or summary_context:
                user_prompt = f"Conversation history:\n{conversation_history}\n\nLatest user message:\n{user_text}"
            else:
                user_prompt = user_text

            # ----------------------------------------------------------
            # VISUAL AIDS
            # ----------------------------------------------------------
            try:
                diagrams_enabled = rag.get_diagram_usage(user_id)
            except Exception:
                diagrams_enabled = False

            # Skip visual aid if only text files attached (speed)
            if file_text_parts and not image_attachments:
                diagrams_enabled = False

            async def _generate_visual_aid():
                if not diagrams_enabled:
                    return None
                try:
                    raw = await ai.get_diagram(
                        site_id=user_id,
                        user=user_text,
                        conversation_context="\n".join(history_lines[-6:]),
                        file_context=file_context,
                        images=image_attachments or None,
                    )
                    if not raw:
                        return None

                    stripped   = raw.strip()
                    first_word = stripped.split(None, 1)[0].upper()

                    if first_word == "NONE":
                        return None

                    if first_word == "MATH":
                        math_body = stripped[len("MATH"):].strip()
                        return {"type": "math", "content": math_body} if math_body else None

                    if first_word == "DIAGRAM":
                        match = re.search(r'<svg.*?</svg>', raw, re.DOTALL | re.IGNORECASE)
                        return {"type": "diagram", "svg": match.group(0)} if match else None

                    if first_word == "CODE":
                        lines = stripped.split("\n", 2)
                        if len(lines) < 3:
                            return None
                        language  = lines[1].strip().lower()
                        code_body = re.sub(r'^```[\w]*\n?', '', lines[2])
                        code_body = re.sub(r'\n?```$', '', code_body).strip("\n")
                        if not language or not code_body:
                            return None
                        return {"type": "code", "language": language, "code": code_body}

                    # Fallback SVG salvage
                    match = re.search(r'<svg.*?</svg>', raw, re.DOTALL | re.IGNORECASE)
                    return {"type": "diagram", "svg": match.group(0)} if match else None

                except Exception as e:
                    print(f"==> Visual aid generation failed: {e}")
                    return None

            if diagrams_enabled:
                try:
                    await ws.send_json({"type": "visual_aid_pending"})
                except Exception:
                    pass

            visual_aid = await _generate_visual_aid()

            # Send visual aid to frontend
            if visual_aid is None:
                try:
                    await ws.send_json({"type": "visual_aid_none"})
                except Exception:
                    pass
            elif visual_aid["type"] == "diagram":
                await ws.send_json({"type": "diagram", "svg": visual_aid["svg"]})
            elif visual_aid["type"] == "code":
                await ws.send_json({"type": "code", "language": visual_aid["language"], "code": visual_aid["code"]})
            elif visual_aid["type"] == "math":
                await ws.send_json({"type": "math", "content": visual_aid["content"]})

            # Save visual to DB
            if visual_aid:
                try:
                    content_to_save = visual_aid.get("svg") or visual_aid.get("code") or visual_aid.get("content", "")
                    rag.save_visual(
                        session_id=chat_id,
                        visual_type=visual_aid["type"],
                        content=content_to_save,
                        language=visual_aid.get("language"),
                    )
                except Exception as e:
                    print(f"==> Failed to save visual: {e}")

            # ----------------------------------------------------------
            # BUILD GROUNDING CONTEXT
            # ----------------------------------------------------------
            visual_aid_summary = ""

            if visual_aid and visual_aid["type"] == "diagram":
                labels = re.findall(r'<text[^>]*>(.*?)</text>', visual_aid["svg"], re.DOTALL | re.IGNORECASE)
                labels = [re.sub(r'\s+', ' ', l).strip() for l in labels if l.strip()]
                if labels:
                    visual_aid_summary = (
                        "A diagram has been shown to the user. It contains: " + ", ".join(labels) + ". "
                        "Speak naturally about the topic consistent with these elements. "
                        "Do not announce the diagram or say 'as you can see'."
                    )

            elif visual_aid and visual_aid["type"] == "code":
                code_preview = visual_aid["code"][:1500]
                visual_aid_summary = (
                    f"A {visual_aid['language']} code snippet has been shown to the user:\n\n{code_preview}\n\n"
                    "Explain what it does conversationally. Don't read it line by line. "
                    "Don't announce that code was shown."
                )

            elif visual_aid and visual_aid["type"] == "math":
                math_preview = visual_aid["content"][:1500]
                visual_aid_summary = (
                    f"A mathematical derivation has been shown to the user:\n\n{math_preview}\n\n"
                    "Give a brief 2-3 sentence spoken summary of what the derivation shows. "
                    "Do not walk through every step. The user can read the working themselves. "
                    "Just tell them the key idea and the result. Keep it under 50 words."
                )

            if visual_aid_summary:
                system_prompt = f"{system_prompt}\n\n{visual_aid_summary}"
            elif visual_aid is None and diagrams_enabled:
                system_prompt = f"{system_prompt}\n\nNo visual aid was needed here. Respond conversationally."
            else:
                system_prompt = (
                    f"{system_prompt}\n\n"
                    "Visual aids are OFF. Answer fully in spoken words. "
                    "For code questions, explain the logic verbally. "
                    "Mention enabling visual aids only if it would genuinely help."
                )

            # File context injection
            if file_context:
                system_prompt = (
                    f"{system_prompt}\n\n"
                    f"The user has attached {len(files_payload)} file(s). Here is the content:\n{file_context}\n\n"
                    "Use this as context. Do not read it verbatim. Explain conversationally."
                )
            else:
                system_prompt = f"{system_prompt}\n\nNo files attached."

            # ----------------------------------------------------------
            # GENERATE RESPONSE
            # ----------------------------------------------------------
            async def _generate_chat():
                sentences      = []
                sentence_buffer = ""
                async for delta in ai.stream(
                    site_id=user_id,
                    system=system_prompt,
                    user=user_prompt,
                    images=image_attachments or None,
                ):
                    sentence_buffer += delta
                    while re.search(r'[.?!,]\s', sentence_buffer):
                        match = re.search(r'[.?!,]\s', sentence_buffer)
                        cut   = match.end()
                        sentence = sentence_buffer[:cut].strip()
                        sentence_buffer = sentence_buffer[cut:]
                        if sentence and len(sentence) > 2:
                            sentences.append(sentence)
                if sentence_buffer.strip() and len(sentence_buffer.strip()) > 2:
                    sentences.append(sentence_buffer.strip())
                return sentences, " ".join(sentences)

            try:
                sentences, bot_text = await _generate_chat()
                bot_text = fix_markdown_formatting(bot_text)
                bot_text  = re.sub(r'```[a-z]*\n?.*?```', '', bot_text, flags=re.DOTALL).strip()
                bot_text  = re.sub(r'\n\s*\n', '\n\n', bot_text)
                sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', bot_text) if len(s.strip()) > 2]

                if not sentences:
                    await ws.send_json({"type": "error", "message": "No response generated"})
                    await ws.send_json({"type": "done"})
                    continue

                # Check if response exceeds TTS length limit
                MAX_BOT_TEXT_LENGTH = 2500  # Define maximum character length for TTS
                text_too_long_for_tts = len(bot_text) > MAX_BOT_TEXT_LENGTH
                
                if text_too_long_for_tts:
                    print(f"==> Bot response too long for TTS: {len(bot_text)} characters (max {MAX_BOT_TEXT_LENGTH}). Sending text only.")
                    audio_on = False  # Disable TTS for this response only

            except Exception as e:
                await ws.send_json({"type": "error", "message": f"AI failed: {str(e)}"})
                continue

            # Send text to frontend regardless of length
            await ws.send_json({"type": "text", "text": bot_text})

            try:
                rag.add_message(chat_id=chat_id, role="assistant", content=bot_text)
                rag.update_last_message(chat_id=chat_id, last_message=bot_text)
            except Exception:
                pass

            try:
                recent_for_summary = history[-6:] if len(history) > 6 else list(history)
                recent_for_summary.append({"role": "user", "content": user_text})
                recent_for_summary.append({"role": "assistant", "content": bot_text})
                await smgr.on_new_message(chat_id, recent_for_summary[-6:])
            except Exception as e:
                print(f"Summary update error: {e}")

            # ----------------------------------------------------------
            # TTS (only if audio_on AND text is not too long)
            # ----------------------------------------------------------
            if audio_on:
                try:
                    cleaned_sentences = [strip_markdown(s) for s in sentences]
                    all_audio, all_visemes, _ = await _run_tts(cleaned_sentences, voice_id)
                    if not all_audio:
                        await ws.send_json({"type": "error", "message": "TTS produced no audio"})
                    else:
                        await _send_audio(ws, all_audio, all_visemes)
                except Exception as e:
                    traceback.print_exc()
                    await ws.send_json({"type": "error", "message": f"TTS failed: {str(e)}"})
            else:
                if text_too_long_for_tts:
                    await ws.send_json({"type": "info", "message": "Response is too long to speak out loud, but you can read it above."})

            # ----------------------------------------------------------
            # COST TRACKING
            # ----------------------------------------------------------
            try:
                billable_visual_text = visual_aid.get("svg") or visual_aid.get("code") or visual_aid.get("content", "") if visual_aid else ""
                file_text_chars = sum(len(p) for p in file_text_parts if not p.startswith("[Image"))

                cost = account_manager.processUsedCost(
                    outputText=bot_text,
                    outputDiagramText=billable_visual_text,
                    inputText=user_prompt,
                    SST_Length_seconds=len(audio_bytes) / 16000 if audio_bytes else 0,
                    webSearch=bool(web_search),
                    voice_on=bool(audio_on),
                    diagram_on=bool(visual_aid),
                    pro_mode=bool(pro_mode),
                    file_text_chars=file_text_chars,
                    image_count=len(image_attachments),
                )
                credits_used = cost / 0.15
                remaining    = rag.deductCredits(user_id, credits_used)
                print(f"==> Cost: ${cost:.4f} | Credits deducted: {credits_used:.4f} | Remaining: {remaining}")
            except Exception as e:
                print(f"==> Cost tracking failed: {e}")

            await ws.send_json({"type": "done"})
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
        
@router.websocket("/embed_chat_ws")
async def embed_chat_ws(ws: WebSocket):
    """
    WebSocket endpoint for embedded widget.
    Auth via API key in query param. Billing per API key and per owner's business credits.
    No diagram generation — embed flow is voice + text only.
    """
    print("HIT embed_chat_ws")
    await ws.accept()

    api_key = ws.query_params.get("api_key")
    if not api_key:
        await ws.close(code=4001, reason="Missing api_key")
        return

    # Initial validation
    initial_key_data = rag.getApiKey(api_key)
    if not initial_key_data or not initial_key_data.get("is_active"):
        await ws.close(code=4001, reason="Invalid or inactive API key")
        return

    # The business credits live on the developer's account (owner_user_id),
    # NOT on the API key itself. We need the owner's UID for every credit check.
    owner_user_id = initial_key_data.get("owner_user_id")
    if not owner_user_id:
        await ws.close(code=4001, reason="API key has no owner")
        return

    # Session ID for conversation threading
    import uuid
    embed_session_id = f"embed_{api_key}_{uuid.uuid4().hex[:12]}"

    FORMATTING_RULE = (
        "CRITICAL FORMATTING RULE: Never use markdown formatting of any kind. "
        "No asterisks, no bold, no headers, no hashtags, no bullet points, no numbered lists, "
        "no dashes, no colons, no semicolons. Write in plain conversational paragraphs only. "
        "This is spoken aloud, not read on screen. "
        "You do NOT have the ability to generate diagrams, charts, or images. "
        "If asked to draw something, politely explain you can only respond in speech."
    )
    SYSTEM_PROMPT = f"""
    {FORMATTING_RULE}
    
    CORE IDENTITY AND PURPOSE
    You are a support agent for {business_name}. {business_description}
    Your role is to provide accurate, helpful support based exclusively on information you have been given.
    
    CRITICAL OPERATING PRINCIPLES
    
    1. KNOWLEDGE BOUNDARY
    - You ONLY answer based on information explicitly provided to you in your knowledge base.
    - You MUST NOT invent, assume, speculate, or extrapolate beyond what you know.
    - You MUST NOT make up product features, policies, pricing, or services.
    - You MUST NOT guess what the company might do, might offer, or might allow.
    - You MUST NOT fill gaps in knowledge with plausible-sounding fabrications.
    
    2. HONESTY AND TRANSPARENCY
    - If you do not know the answer, say so immediately and directly.
    - Examples of honest responses:
        "I don't have that information in my knowledge base."
        "That's not something I can answer based on what I know about {business_name}."
        "I'm not sure about that. Let me suggest contacting [support method] for accurate details."
    - Never pretend to know something. Uncertainty is acceptable. Fabrication is not.
    
    3. SCOPE BOUNDARIES
    - Stick to questions about {business_name}, its services, and its policies.
    - If asked about competitors, other businesses, or unrelated topics, politely redirect.
    - Example: "I'm specifically here to help with {business_name}. I can't speak to other companies."
    - Do not attempt to answer general knowledge questions unless directly related to {business_name}.
    
    4. TONE AND STYLE
    - Be conversational, warm, and helpful.
    - Keep responses under 100 words unless the user explicitly asks for more detail.
    - Do not sound robotic or overly formal.
    - Speak as if you are having a natural conversation.
    - Use "I" statements: "I don't have that information" not "this agent cannot determine."
    
    5. RESPONSE STRUCTURE FOR UNKNOWNS
    - Acknowledge the question: "That's a great question."
    - Be honest about your limitation: "I don't have that specific detail."
    - Offer a path forward: "You could reach out to [contact method] for the most accurate answer."
    - Never just say no without offering an alternative.
    
    6. INFORMATION VERIFICATION
    - Before answering about policies, features, or details, verify it matches your knowledge base.
    - If a user claims something about {business_name} that you cannot verify, do not confirm it.
    - Example: "I don't have that information confirmed, so I can't say for certain."
    
    7. PROHIBITED BEHAVIORS
    - Do NOT use phrases like "Based on my training" or "I believe" when discussing {business_name} specifics.
    - Do NOT make up support contact information, email addresses, or phone numbers.
    - Do NOT promise outcomes you cannot guarantee.
    - Do NOT create fictional policies, discounts, or exceptions.
    - Do NOT roleplay as multiple people or departments.
    - Do NOT offer legal, financial, or medical advice even if loosely related to {business_name}.
    
    8. WHEN IN DOUBT
    - Err on the side of honesty over helpfulness.
    - If a question touches on something you're not 100% certain about, acknowledge the uncertainty.
    - Suggest the user verify with an official channel: sales, support, management, or documentation.
    
    9. HANDLING EDGE CASES
    - Vague questions: Ask for clarification. "Could you tell me more about what you're looking for?"
    - Multi-part questions: Answer what you know, be honest about what you don't.
    - Hypothetical questions: "I can only speak to what {business_name} currently does."
    - Questions about the future: "I don't have information about planned changes."
    
    10. RESPONSE LENGTH AND CLARITY
        - Keep responses conversational and under 100 words unless asked for detail.
        - If detail is requested, expand your answer but stay grounded in your knowledge base.
        - Use short sentences for audio clarity.
        - Pause naturally where you might take a breath in speech.
    
    KNOWLEDGE BASE REFERENCE
    The following information has been verified and you may use it confidently:
    - Services offered: [INSERT ACTUAL SERVICES]
    - Policies: [INSERT ACTUAL POLICIES]
    - Common questions: [INSERT ACTUAL FAQs]
    - Contact information: [INSERT ACTUAL CONTACT INFO]
    - Pricing (if applicable): [INSERT ACTUAL PRICING]
    
    ANYTHING NOT IN THE ABOVE LIST IS OUT OF BOUNDS.
    
    BEGIN CONVERSATION
    You are now ready to assist. Remember: accuracy and honesty are your top priorities. 
    Start each interaction fresh and ask clarifying questions if needed.
    """
    # Minimum credits required to start a turn. A typical turn costs 1-5 cents
    # depending on length and whether TTS is enabled, so we require at least
    # 5 cents to begin. This prevents starting a turn we can't afford to finish.
    MIN_CREDITS_PER_TURN = 0.05  # 5 cents AUD

    print(f"==> embed WS opened for api_key={api_key}, owner={owner_user_id}, session={embed_session_id}")

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

            # ----------------------------------------------------------
            # RE-VALIDATE API KEY (catches dashboard toggles mid-session)
            # ----------------------------------------------------------
            key_data = rag.getApiKey(api_key)
            if not key_data or not key_data.get("is_active"):
                await ws.send_json({"type": "error", "message": "API key deactivated", "code": "INVALID_KEY"})
                await ws.close()
                return

            # ----------------------------------------------------------
            # CONVERSATION LIMIT CHECK (monthly cap per API key)
            # ----------------------------------------------------------
            if key_data.get("conversations_used", 0) >= key_data.get("monthly_limit", 500):
                await ws.send_json({
                    "type": "error",
                    "message": "Monthly conversation limit reached",
                    "code": "LIMIT_REACHED"
                })
                await ws.send_json({"type": "done"})
                continue

            # ----------------------------------------------------------
            # BUSINESS CREDIT CHECK (owner's wallet balance)
            # ----------------------------------------------------------
            # Before running the turn, make sure the owner has enough credits
            # to plausibly pay for it. This prevents giving away free turns to
            # accounts that have run dry.
            try:
                current_credits = rag.getBusinessCredits(owner_user_id)
                if current_credits < MIN_CREDITS_PER_TURN:
                    await ws.send_json({
                        "type": "error",
                        "message": "This business has run out of credits. Please contact them to continue the conversation.",
                        "code": "NO_CREDITS"
                    })
                    await ws.send_json({"type": "done"})
                    continue
            except Exception as e:
                print(f"==> Credit check failed: {e}")
                # If the check itself fails (DB issue), fail closed to protect the owner's wallet
                await ws.send_json({
                    "type": "error",
                    "message": "Temporary billing error. Please try again shortly.",
                    "code": "BILLING_ERROR"
                })
                await ws.send_json({"type": "done"})
                continue

            # ----------------------------------------------------------
            # DAILY COST CAP (per API key, optional belt and suspenders)
            # ----------------------------------------------------------
            try:
                daily_cost = rag.getApiKeyDailyCost(api_key)
                daily_cap = key_data.get("daily_cost_cap", 10.00)  # $10 default
                if daily_cost >= daily_cap:
                    await ws.send_json({
                        "type": "error",
                        "message": "Daily usage cap reached. Try again tomorrow.",
                        "code": "DAILY_CAP"
                    })
                    await ws.send_json({"type": "done"})
                    continue
            except Exception:
                pass  # Fail open if daily cap check errors — main credit check already protects us

            # ----------------------------------------------------------
            # EXTRACT PAYLOAD
            # ----------------------------------------------------------
            business_name = key_data.get("business_name") or ""
            business_description = key_data.get("business_description") or ""
            avatar_name = key_data.get("avatar_name") or "Mia Sterling"
            personality_on = bool(key_data.get("personality_on"))

            user_text = _as_str(payload.get("message"))
            voice_id = _as_str(payload.get("voice_name"))
            audio_on = bool(payload.get("voice_on", True))
            raw_audio = payload.get("audio_bytes")

            session_id = _as_str(payload.get("session_id"))

            if raw_audio and "," in raw_audio:
                raw_audio = raw_audio.split(",", 1)[1]
            audio_bytes = base64.b64decode(raw_audio) if raw_audio else None

            # ----------------------------------------------------------
            # STT (if audio was sent)
            # ----------------------------------------------------------
            if audio_bytes:
                try:
                    stt_instance = get_stt()
                    user_text = stt_instance.get_transcript(audio_bytes)
                    if not user_text:
                        await ws.send_json({"type": "error", "message": "Could not understand audio."})
                        await ws.send_json({"type": "done"})
                        continue
                    await ws.send_json({"type": "transcript", "text": user_text})
                except Exception as e:
                    await ws.send_json({"type": "error", "message": f"Transcription failed: {e}"})
                    await ws.send_json({"type": "done"})
                    continue

            if not user_text:
                await ws.send_json({"type": "error", "message": "Missing message"})
                await ws.send_json({"type": "done"})
                continue

            # ----------------------------------------------------------
            # LOAD AVATAR + VOICE
            # ----------------------------------------------------------
            avatar = rag.getAvatarByName(avatar_name) or {}
            if not voice_id:
                voice_id = avatar.get("voice") or "UgBBYS2sOqTuMpoF3BR0"

            # ----------------------------------------------------------
            # CONVERSATION HISTORY
            # ----------------------------------------------------------
            try:
                history = rag.get_embed_messages_by_session(
                    session_id=session_id,
                    limit=10,
                )
            except Exception:
                history = []

            try:
                rag.add_embed_message(
                    session_id=session_id,
                    api_key=api_key,
                    role="user",
                    content=user_text,
                )
            except Exception as e:
                print(f"==> Failed to save user message: {e}")

            # ----------------------------------------------------------
            # RAG LOOKUP (business knowledge base)
            # ----------------------------------------------------------
            rag_context = ""
            try:
                embedding = await rag.embedText(user_text)
                chunks = rag.searchDocumentChunks(api_key=api_key, embedding=embedding, limit=5)
                if chunks and chunks[0].get("similarity", 0) >= 0.3:
                    rag_context = "\n".join(f"- {c['content']}" for c in chunks)
            except Exception as e:
                print(f"==> RAG failed: {e}")

            # ----------------------------------------------------------
            # BUILD SYSTEM PROMPT
            # ----------------------------------------------------------
            if personality_on:
                avatar_prompt = avatar.get("prompt") or ""
                fallback_prompt = key_data.get("system_prompt") or ""
                persona = avatar_prompt or fallback_prompt
                system_prompt = f"{FORMATTING_RULE}\n\n{persona}"
            else:
                system_prompt = SYSTEM_PROMPT

            if rag_context:
                system_prompt = f"{system_prompt}\n\nRelevant information from the business knowledge base:\n{rag_context}"

            # ----------------------------------------------------------
            # BUILD USER PROMPT WITH HISTORY
            # ----------------------------------------------------------
            history_lines = []
            for m in history[-6:]:
                role = (m.get("role") or "").lower()
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if role == "user":
                    history_lines.append(f"User: {content}")
                elif role == "assistant":
                    history_lines.append(f"Assistant: {content}")

            if history_lines:
                user_prompt = (
                    "Conversation history:\n"
                    + "\n".join(history_lines)
                    + f"\n\nLatest user message:\n{user_text}"
                )
            else:
                user_prompt = user_text

            # ----------------------------------------------------------
            # STREAM GEMINI
            # ----------------------------------------------------------
            try:
                sentences = []
                sentence_buffer = ""
                async for delta in ai.stream(site_id=api_key, system=system_prompt, user=user_prompt):
                    sentence_buffer += delta
                    while (match := re.search(r'[.?!]\s', sentence_buffer)):
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

                # Strip any stray code fences the model might sneak in
                bot_text = re.sub(r'```[a-z]*\n?.*?```', '', bot_text, flags=re.DOTALL).strip()
                bot_text = re.sub(r'\n\s*\n', '\n\n', bot_text)

                # Rebuild sentences from cleaned text
                sentences = [
                    s.strip()
                    for s in re.split(r'(?<=[.?!])\s+', bot_text)
                    if len(s.strip()) > 2
                ]

                print(f"==> embed {len(sentences)} sentences", flush=True)

            except Exception as e:
                await ws.send_json({"type": "error", "message": "The assistant is unavailable right now. Please try again shortly."})
                print(f"==> AI failed: {e}")
                await ws.send_json({"type": "done"})
                continue

            # ----------------------------------------------------------
            # SEND TEXT TO CLIENT
            # ----------------------------------------------------------
            await ws.send_json({"type": "text", "text": bot_text})

            # ----------------------------------------------------------
            # SAVE ASSISTANT RESPONSE
            # ----------------------------------------------------------
            try:
                rag.add_embed_message(
                    session_id=embed_session_id,
                    api_key=api_key,
                    role="assistant",
                    content=bot_text,
                )
            except Exception as e:
                print(f"==> Failed to save assistant message: {e}")

            # ----------------------------------------------------------
            # INCREMENT CONVERSATION COUNT
            # ----------------------------------------------------------
            try:
                rag.incrementConversationCount(api_key)
            except Exception as e:
                print(f"==> Failed to increment conversation count: {e}")

            # ----------------------------------------------------------
            # TTS (if audio enabled)
            # ----------------------------------------------------------
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
                    await ws.send_json({"type": "error", "message": "Voice playback failed."})

            # ----------------------------------------------------------
            # COST TRACKING + CREDIT DEDUCTION
            # This is the part that was broken before. Single try/except,
            # and it now actually deducts from the owner's business credits.
            # ----------------------------------------------------------
            try:
                cost_aud = account_manager.processUsedCost(
                    outputText=bot_text,
                    outputDiagramText="",
                    inputText=user_prompt,
                    SST_Length_seconds=len(audio_bytes) / 32000 if audio_bytes else 0,  # 16kHz 16-bit = 32000 bytes/sec
                    webSearch=False,
                    voice_on=bool(audio_on),
                    diagram_on=False,
                )

                # Track per-API-key cost (for daily cap and analytics)
                try:
                    rag.addApiKeyCost(api_key, cost_aud)
                except Exception as e:
                    print(f"==> Failed to add api_key cost: {e}")

                # Deduct from the owner's business credits wallet
                try:
                    new_balance = rag.deductBusinessCredits(owner_user_id, cost_aud)
                    print(
                        f"==> Embed cost: ${cost_aud:.4f} AUD deducted from owner={owner_user_id}, "
                        f"new balance: ${new_balance:.4f}",
                        flush=True,
                    )

                    # Warn the client if the owner is running low (below 50 cents)
                    # so they can show a "running low" UI or notify the business
                    if new_balance < 0.50:
                        await ws.send_json({
                            "type": "credits_low",
                            "balance": new_balance,
                            "message": "Credits running low"
                        })
                except Exception as e:
                    print(f"==> Failed to deduct business credits: {e}")

            except Exception as e:
                print(f"==> Embed cost tracking failed: {e}", flush=True)

            # ----------------------------------------------------------
            # SIGNAL TURN COMPLETE
            # ----------------------------------------------------------
            try:
                await ws.send_json({"type": "done"})
                print("==> DONE sent, turn complete", flush=True)
            except Exception as e:
                print(f"==> Failed to send done: {e}", flush=True)

    except WebSocketDisconnect:
        return
    except Exception as e:
        print("❌ Embed WS error:", repr(e))
        traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "message": "Connection error"})
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass