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
import json
from Providers.microsoft_auth import *
summary_mgr = None
rag = VectorRAGService()
ai = AIProvider(rag)
account_manager = AccountManager(rag)
fileE = FileExtractor(ai)
from datetime import datetime, timezone
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
    # --- HEADERS ---
    # Ensure blank line BEFORE headers
    text = re.sub(r'(?<!\n)\n(#{1,6}\s)', r'\n\n\1', text)
    
    # Split header runs into header + body when smooshed on one line
    text = re.sub(r'^(#{1,6}\s+[^\n]+?[?.!])\s+(?=[A-Z])', r'\1\n\n', text, flags=re.MULTILINE)
    
    def split_long_header(match):
        hashes, content = match.group(1), match.group(2)
        if len(content) < 60:
            return f"{hashes} {content}"
        parts = re.split(r'(?<=[.?!])\s+', content, maxsplit=1)
        if len(parts) == 2:
            return f"{hashes} {parts[0]}\n\n{parts[1]}"
        return f"{hashes} {content}"
    
    text = re.sub(r'^(#{1,6})\s+([^\n]+)$', split_long_header, text, flags=re.MULTILINE)
    text = re.sub(r'(^#{1,6}\s+[^\n]+)\n(?!\n)', r'\1\n\n', text, flags=re.MULTILINE)

    # --- BULLET LISTS ---
    # Force a newline before any inline "*   " or "-   " bullet that isn't already at line start
    # Matches: ". *   Websites:" or "content. * Automation:"
    text = re.sub(r'(?<=[.!?:])\s+(\*\s{2,}\*\*)', r'\n\1', text)
    text = re.sub(r'(?<=[.!?:])\s+(-\s{2,}\*\*)', r'\n\1', text)
    
    # Also handle single-space bullets like "* **Item**"
    text = re.sub(r'(?<=[.!?:])\s+(\*\s+\*\*[A-Z])', r'\n\1', text)

    # --- NUMBERED LISTS ---
    # The model often writes "1. **Writing:** ... **Translation:** ... **Execution:**"
    # We need to detect bold-prefixed items that should be numbered list continuations.
    # Heuristic: if we're inside a numbered list context and see ". **Word:**" inline, split it.
    def fix_numbered_list_run(match):
        full = match.group(0)
        # Split on ". **" boundaries that look like new list items
        parts = re.split(r'(?<=[.!?])\s+(?=\*\*[A-Z][a-zA-Z]+:?\*\*)', full)
        if len(parts) <= 1:
            return full
        # Renumber sequentially
        first_num_match = re.match(r'^(\d+)\.\s+', parts[0])
        if not first_num_match:
            return full
        start_num = int(first_num_match.group(1))
        result = [parts[0]]
        for i, part in enumerate(parts[1:], start=1):
            result.append(f"{start_num + i}. {part}")
        return "\n".join(result)
    
    # Apply to lines that start with "N. **Something**" and contain more bold runs
    text = re.sub(
        r'^\d+\.\s+\*\*[^*]+\*\*[^\n]*(?:\s+\*\*[^*]+\*\*[^\n]*)+',
        fix_numbered_list_run,
        text,
        flags=re.MULTILINE
    )

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
async def _tts_pipeline(ws, sentences_queue: asyncio.Queue, voice_id: str, done_event: asyncio.Event):
    """
    Consumes sentences from the queue, fires TTS, and sends audio chunks
    to the client as they complete. Runs as a background task.
    
    Sends one audio_begin at the start, streams chunks, sends audio_end when done.
    """
    tts_instance = get_tts()
    cumulative_offset_ms = 0
    first_chunk = True
    all_visemes = []
    
    while True:
        # Wait for a sentence or the done signal
        try:
            sentence = await asyncio.wait_for(sentences_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            if done_event.is_set() and sentences_queue.empty():
                break
            continue
        
        if sentence is None:  # Poison pill
            break
        
        try:
            ROUTING_TAGS = {'none', 'inline', 'diagram', 'code', 'math', 'html'}
            cleaned = strip_markdown(sentence)
            if not cleaned or len(cleaned.strip()) < 3:
                continue
            if cleaned.strip().lower() in ROUTING_TAGS:
                continue
            cleaned = re.sub(r'\[?(NONE|INLINE|DIAGRAM|CODE|MATH|HTML)\]?\s*$', '', cleaned, flags=re.IGNORECASE).strip()
            if not cleaned or len(cleaned) < 3:
                continue
            result = await tts_instance.synthesize_sentence(cleaned, voice_id)
            if isinstance(result, Exception):
                print(f"==> TTS sentence failed: {result}", flush=True)
                continue
                
            audio_bytes_chunk, visemes, duration = result
            
            if not audio_bytes_chunk:
                continue
            
            # Offset visemes
            chunk_visemes = []
            for v in visemes:
                chunk_visemes.append({
                    "t_ms": v["t_ms"] + cumulative_offset_ms,
                    "viseme_id": v["viseme_id"],
                })
            
            if first_chunk:
                # Send audio_begin with first batch of visemes
                await ws.send_json({
                    "type": "audio_begin",
                    "format": "mp3",
                    "sample_rate_hz": 44100,
                    "channels": 1,
                    "visemes": [
                        {"t_ms": _as_int(v.get("t_ms"), 0), "viseme_id": _as_int(v.get("viseme_id"), 0)}
                        for v in chunk_visemes
                    ],
                })
                first_chunk = False
            else:
                # Send additional visemes for subsequent chunks
                if chunk_visemes:
                    await ws.send_json({
                        "type": "viseme_update",
                        "visemes": [
                            {"t_ms": _as_int(v.get("t_ms"), 0), "viseme_id": _as_int(v.get("viseme_id"), 0)}
                            for v in chunk_visemes
                        ],
                    })
            
            # Send audio bytes
            CHUNK = 32_000
            for i in range(0, len(audio_bytes_chunk), CHUNK):
                await ws.send_bytes(audio_bytes_chunk[i:i + CHUNK])
            
            cumulative_offset_ms += int(duration * 1000)
            all_visemes.extend(chunk_visemes)
            
        except Exception as e:
            print(f"==> TTS pipeline error: {e}", flush=True)
            traceback.print_exc()
    
    # Send audio_end if we sent any audio
    if not first_chunk:
        try:
            await ws.send_json({"type": "audio_end"})
        except Exception:
            pass
 
def _extract_routing_tag(text: str) -> tuple:
    """
    Looks for [NONE], [INLINE], [DIAGRAM], [CODE], or [MATH] at the end of the text.
    Returns (cleaned_text, routing_tag).
    """
    import re
    # Match routing tag at end of text (possibly with trailing whitespace)
    match = re.search(r'\[(NONE|INLINE|DIAGRAM|CODE|MATH|HTML)\]\s*$', text, re.IGNORECASE)
    if match:
        tag = match.group(1).upper()
        cleaned = text[:match.start()].rstrip()
        return cleaned, tag
    
    # Fallback: check last line
    lines = text.strip().rsplit('\n', 1)
    if len(lines) == 2:
        last_line = lines[1].strip().upper()
        if last_line in ('NONE', 'INLINE', 'DIAGRAM', 'CODE', 'MATH', 'HTML',
                 '[NONE]', '[INLINE]', '[DIAGRAM]', '[CODE]', '[MATH]', '[HTML]'):
            tag = last_line.strip('[]')
            return lines[0].rstrip(), tag
    
    # No tag found — default to NONE
    return text, "NONE"
 
 
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
# CHAT DIAGRAM INIT
# -----------------------------------------------------------------------

@router.post("/chat_diagram_init")
async def chat_diagram_init(init_details: DiagramInit, user=Depends(verify_token)):
    chatID = init_details.chat_id

    raw_visuals = rag.get_visuals(chatID)
    
    visuals = []
    for v in raw_visuals:
        visuals.append({
            "visual_type": v.get("visual_type"),
            "content": v.get("content"),
            "language": v.get("language"),
            "created_at": str(v.get("created_at", "")),
        })
    print(f"=> VISUALS {visuals} {raw_visuals}")
    return {"visuals": visuals}


# -----------------------------------------------------------------------
# FOR NORMAL CHATTING
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
            chat_input_tokens = 0 
            chat_output_tokens = 0
            diagram_input_tokens = 0
            diagram_output_tokens = 0



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

            recent_history = history[-100:] if len(history) < 100 else history

            if web_search:
                web_response = get_web_search().web_search(user_text, 3)
                system_prompt = f"{system_prompt}\n\n{web_response}"

            MAX_INPUT_CHARS = 30000

            history_lines = []
            for m in recent_history:
                role    = (m.get("role") or "").lower()
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if role == "user":
                    if len(content) > MAX_INPUT_CHARS:
                        content = content[:MAX_INPUT_CHARS]
                    history_lines.append(f"User: {content}")
                elif role == "assistant":
                    history_lines.append(f"Assistant: {content}")

            conversation_history = "\n".join(history_lines)
            if conversation_history or summary_context:
                user_prompt = f"Conversation history:\n{conversation_history}\n\nLatest user message:\n{user_text}"
            else:
                user_prompt = user_text

            
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
            # LENGTH RULE
            # ----------------------------------------------------------
            if audio_on:
                system_prompt += (
                    "\n\nLENGTH RULE: This response will be spoken aloud. Keep it under 120 words. "
                    "Lead with the core answer, then the most important detail. "
                    "If the topic needs a visual, keep your spoken response to a brief summary."
                )
            else:
                system_prompt += (
                    "\n\nLENGTH RULE: Audio is off. You have room to be thorough. "
                    "Use headers, lists, and examples freely."
                )
            # ----------------------------------------------------------
            # GENERATE RESPONSE (streaming to client + per-sentence TTS)
            # ----------------------------------------------------------
            try:
                full_text_parts = []
                sentence_buffer = ""
                sentences_for_tts = []  # Keep track for cost/fallback
                tts_queue = None
                tts_task = None
                tts_done_event = None
                
                # Set up TTS pipeline if audio is on
                if audio_on:
                    tts_queue = asyncio.Queue()
                    tts_done_event = asyncio.Event()
                    tts_task = asyncio.create_task(
                        _tts_pipeline(ws, tts_queue, voice_id, tts_done_event)
                    )
                
                # Track if we've hit the routing tag
                routing_tag_found = False
                visual_section_buffer = ""
 
                async for delta in ai.stream(
                    site_id=user_id,
                    system=system_prompt,
                    user=user_prompt,
                    images=image_attachments or None,
                ):
                    if delta.startswith("__USAGE__"):
                        try:
                            parts = delta[len("__USAGE__"):].split(",")
                            chat_input_tokens  += int(parts[0])
                            chat_output_tokens += int(parts[1])
                        except Exception:
                            pass
                        continue
                    full_text_parts.append(delta)
                    
                    # Stream to client (text appears immediately)
                    await ws.send_json({"type": "text_delta", "text": delta})

                    # Accumulate for sentence detection
                    sentence_buffer += delta
                    while re.search(r'[.?!]\s', sentence_buffer):
                        match = re.search(r'[.?!]\s', sentence_buffer)
                        cut = match.end()
                        sentence = sentence_buffer[:cut].strip()
                        sentence_buffer = sentence_buffer[cut:]
                        if sentence and len(sentence) > 2:
                            sentences_for_tts.append(sentence)
                            # Fire TTS immediately for this sentence
                            if tts_queue and not routing_tag_found:
                                await tts_queue.put(sentence)

                # Handle remaining buffer
                remaining = sentence_buffer.strip()
                # Strip routing tag before sending to TTS
                remaining = re.sub(r'\[?(NONE|INLINE|DIAGRAM|CODE|MATH|HTML)\]?\s*$', '', remaining, flags=re.IGNORECASE).strip()
                if remaining and len(remaining) > 2:
                    sentences_for_tts.append(remaining)
                    if tts_queue:
                        await tts_queue.put(remaining)
 
                # Signal TTS pipeline that no more sentences are coming
                if tts_done_event:
                    tts_done_event.set()
 
                # Join full text and extract routing tag
                raw_text = "".join(full_text_parts)
                print(f"==> RAW TAIL: {repr(raw_text[-80:])}", flush=True)
                bot_text, routing_decision = _extract_routing_tag(raw_text)
                print(f"==> ROUTING: {routing_decision}", flush=True)
                # Clean up markdown
                bot_text = fix_markdown_formatting(bot_text)
                tts_text  = re.sub(r'```[a-z]*\n?.*?```', '', bot_text, flags=re.DOTALL).strip()
                tts_text  = re.sub(r'\n\s*\n', '\n\n', tts_text)
 
                # Send final cleaned version
                await ws.send_json({"type": "text_done", "text": bot_text})
 
                # Build sentences list for any remaining processing
                parts = re.split(r'(#{1,6}\s+[^\n]+)', tts_text)
                sentences = []
                for part in parts:
                    if re.match(r'^#{1,6}\s+', part):
                        sentences.append(part.strip())
                    else:
                        sub_sentences = re.split(r'(?<=[.?!])\s+', part)
                        sentences.extend([s.strip() for s in sub_sentences if len(s.strip()) > 2])
 
                if not sentences and not sentences_for_tts:
                    await ws.send_json({"type": "error", "message": "No response generated"})
                    await ws.send_json({"type": "done"})
                    if tts_task:
                        tts_task.cancel()
                    continue
 
                print(f"==> Routing decision: {routing_decision}", flush=True)
 
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"AI failed: {str(e)}"})
                if tts_task:
                    tts_task.cancel()
                continue
 
            # ----------------------------------------------------------
            # VISUAL AID (Call 2 — fires in parallel with TTS)
            # ----------------------------------------------------------
            visual_aid = None
            visual_task = None

            if routing_decision in ("DIAGRAM", "CODE", "MATH", "HTML"):
                # Fire visual generation while TTS is still playing
                async def _generate_visual_content():
                    nonlocal diagram_input_tokens, diagram_output_tokens
                    try:
                        raw, d_in, d_out = await ai.get_diagram(
                            site_id=user_id,
                            user=user_text,
                            conversation_context="\n".join(history_lines[-6:]),
                            file_context=file_context,
                            images=image_attachments or None,
                        )
                        diagram_input_tokens = d_in
                        diagram_output_tokens = d_out

                        if not raw:
                            return None
                        stripped = raw.strip()
                        stripped = re.sub(r'^```\w*\n?', '', stripped)
                        stripped = re.sub(r'\n?```$', '', stripped)
                        upper = stripped.upper()
 
                        if upper.startswith("NONE") or upper.startswith("INLINE"):
                            return None
 
                        if upper.startswith("MATH"):
                            math_body = stripped[4:].strip().lstrip(":").strip()
                            return {"type": "math", "content": math_body} if math_body else None
 
                        if upper.startswith("DIAGRAM"):
                            match = re.search(r'<svg.*?</svg>', stripped, re.DOTALL | re.IGNORECASE)
                            return {"type": "diagram", "svg": match.group(0)} if match else None
 
                        if upper.startswith("CODE"):
                            body = stripped[4:].strip().lstrip(":").strip()
                            lines = body.split("\n", 1)
                            if len(lines) < 2:
                                return None
                            language = lines[0].strip().lower().replace("`", "")
                            code_body = lines[1]
                            code_body = re.sub(r'^```\w*\n?', '', code_body)
                            code_body = re.sub(r'\n?```$', '', code_body).strip("\n")
                            if not language or not code_body:
                                return None
                            return {"type": "code", "language": language, "code": code_body}
                        if upper.startswith("HTML"):
                            body = stripped[4:].strip().lstrip(":").strip()
                            return {"type": "html", "content": body} if body else None
                        # Fallback SVG salvage
                        match = re.search(r'<svg.*?</svg>', stripped, re.DOTALL | re.IGNORECASE)
                        return {"type": "diagram", "svg": match.group(0)} if match else None
 
                    except Exception as e:
                        print(f"==> Visual aid generation failed: {e}")
                        return None
 
                visual_task = asyncio.create_task(_generate_visual_content())
                # Tell frontend visual is loading
                await ws.send_json({"type": "visual_aid_pending"})
            else:
                # NONE or INLINE — no visual panel
                try:
                    await ws.send_json({"type": "visual_aid_none"})
                except Exception:
                    pass
 
            # ----------------------------------------------------------
            # WAIT FOR TTS TO FINISH
            # ----------------------------------------------------------
            if tts_task:
                try:
                    await asyncio.wait_for(tts_task, timeout=30)
                except asyncio.TimeoutError:
                    tts_task.cancel()
                    print("==> TTS task timed out, cancelled")
                except Exception as e:
                    print(f"==> TTS pipeline task error: {e}", flush=True)
 
            # ----------------------------------------------------------
            # WAIT FOR VISUAL AID + SEND TO FRONTEND
            # ----------------------------------------------------------
            if visual_task:
                try:
                    visual_aid = await visual_task
                except Exception as e:
                    print(f"==> Visual task error: {e}", flush=True)
                    visual_aid = None
 
            # Send visual aid to frontend
            if visual_aid is None and routing_decision in ("DIAGRAM", "CODE", "MATH", "HTML"):
                try:
                    await ws.send_json({"type": "visual_aid_none"})
                except Exception:
                    pass
            elif visual_aid and visual_aid["type"] == "diagram":
                await ws.send_json({"type": "diagram", "svg": visual_aid["svg"]})
            elif visual_aid and visual_aid["type"] == "code":
                await ws.send_json({"type": "code", "language": visual_aid["language"], "code": visual_aid["code"]})
            elif visual_aid and visual_aid["type"] == "math":
                await ws.send_json({"type": "math", "content": visual_aid["content"]})
            elif visual_aid and visual_aid["type"] == "html":
                            await ws.send_json({"type": "html", "content": visual_aid["content"]})
            # Save visual to DB
            if visual_aid and visual_aid.get("type") not in (None, "inline"):
                try:
                    content_to_save = visual_aid.get("svg") or visual_aid.get("code") or visual_aid.get("content", "") or visual_aid.get("math") or visual_aid.get("html")
                    rag.save_visual(
                        session_id=chat_id,
                        visual_type=visual_aid["type"],
                        content=content_to_save,
                        language=visual_aid.get("language"),
                    )
                except Exception as e:
                    print(f"==> Failed to save visual: {e}")
 
            # ----------------------------------------------------------
            # SAVE TO DB
            # ----------------------------------------------------------
            try:
                rag.add_message(chat_id=chat_id, role="assistant", content=bot_text)
                rag.update_last_message(chat_id=chat_id, last_message=bot_text)
            except Exception:
                pass
            # ----------------------------------------------------------
            # Summary Builder
            # ----------------------------------------------------------
            try:
                char_count = 0
                cutoff = len(history)  # start assuming we use all of it

                for j in range(len(history) - 1, -1, -1):
                    char_count += len(history[j].get("content", ""))
                    if char_count > 800_000:
                        cutoff = j + 1  # everything from here forward is within budget
                        break


                trimmed_history = history[cutoff:]
                recent_for_summary = trimmed_history.copy()
                recent_for_summary.append({"role": "user", "content": user_text})
                recent_for_summary.append({"role": "assistant", "content": bot_text})
                await smgr.on_new_message(chat_id, recent_for_summary[-6:])
            except Exception as e:
                print(f"Summary update error: {e}")
            # ---------------------------------------------------------
            # MODEL 
            # ----------------------------------------------------------
            model = rag.get_model(user_id)
            # COST TRACKING
            # ----------------------------------------------------------
            try:
                billable_visual_text = ""
                if visual_aid:
                    billable_visual_text = visual_aid.get("svg") or visual_aid.get("code") or visual_aid.get("math") or visual_aid.get("html") or visual_aid.get("content", "")

                cost = account_manager.processUsedCost(
                    # Call 1 — real tokens
                    input_tokens=chat_input_tokens+diagram_input_tokens,
                    output_tokens=chat_output_tokens+diagram_output_tokens,
                    # Everything else
                    SST_Length_seconds=len(audio_bytes) / 16000 if audio_bytes else 0,
                    webSearch=bool(web_search),
                    voice_on=bool(audio_on),
                    diagram_on=bool(visual_aid),
                    pro_mode=bool(pro_mode),
                    image_count=len(image_attachments),
                    model=model
                )
                credits_used = cost / 0.15
                remaining = rag.deductCredits(user_id, credits_used)
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

# -----------------------------------------------------------------------
# FOR AGENT MODE (BOOKINGS)
# -----------------------------------------------------------------------

@router.websocket("/audio_chat_ws")
async def agent_chat_ws(ws: WebSocket):
    print("HIT agent_chat_ws")
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
            integrations = True

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
            chat_input_tokens = 0 
            chat_output_tokens = 0
            diagram_input_tokens = 0
            diagram_output_tokens = 0


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

            recent_history = history[-100:] if len(history) < 100 else history

            if web_search:
                web_response = get_web_search().web_search(user_text, 3)
                system_prompt = f"{system_prompt}\n\n{web_response}"

            MAX_INPUT_CHARS = 30000

            history_lines = []
            for m in recent_history:
                role    = (m.get("role") or "").lower()
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if role == "user":
                    if len(content) > MAX_INPUT_CHARS:
                        content = content[:MAX_INPUT_CHARS]
                    history_lines.append(f"User: {content}")
                elif role == "assistant":
                    history_lines.append(f"Assistant: {content}")

            conversation_history = "\n".join(history_lines)
            if conversation_history or summary_context:
                user_prompt = f"Conversation history:\n{conversation_history}\n\nLatest user message:\n{user_text}"
            else:
                user_prompt = user_text

            
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
            # LENGTH RULE
            # ----------------------------------------------------------
            if audio_on:
                system_prompt += (
                    "\n\nLENGTH RULE: This response will be spoken aloud. Keep it under 120 words. "
                    "Lead with the core answer, then the most important detail. "
                    "If the topic needs a visual, keep your spoken response to a brief summary."
                )
            else:
                system_prompt += (
                    "\n\nLENGTH RULE: Audio is off. You have room to be thorough. "
                    "Use headers, lists, and examples freely."
                )
                
            # ----------------------------------------------------------
            # INTEGRATIONS CHECK (check if there is integrations enabled for this User)
            # ----------------------------------------------------------
            integrators = rag.checkIntegrations(user_id) #MVP, we force the user to choose a specific integrator. Which one to use. Later one we can create a loop where the AI asks which one the user would like to use
            print(f"==> INTEGRATIONS CHECK: {integrators} for {user_id}")
            if integrators: 
                #Then we add logic to the AI to say hey, these are the integrations the user has, if needed, ask the user what integrator they would like to use
                system_prompt += (
                    "\n\nINTEGRATIONS: The user has connected Microsoft Calendar. "
                    "If the user asks to book, schedule, or create an event/meeting/appointment, respond naturally confirming what you're doing, then at the very end of your response append a calendar action in this exact format with no space between the tag and JSON:\n"
                    "[CALENDAR_WRITE]{\"subject\": \"<title>\", \"start\": \"<ISO datetime>\", \"end\": \"<ISO datetime>\", \"body\": \"<optional notes>\", \"attendees\": []}\n"
                    "If the user asks to check, view, or list their calendar/events/schedule, respond naturally then append:\n"
                    "[CALENDAR_READ]{\"days_ahead\": 7}\n"
                    "IMPORTANT: Only append the tag if a calendar action is clearly needed. For normal conversation, do not append any calendar tag."
                )
            else:
                system_prompt += (
                    "\n\nINTEGRATIONS RULE: User has not yet connected services to the AI for you to be able to perform calendar actions"
                )

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            system_prompt += (
                f"\n\nToday's date is {today} (UTC)."
            )
            # ----------------------------------------------------------
            # GENERATE RESPONSE (streaming to client + per-sentence TTS)
            # ----------------------------------------------------------
            try: 
                full_text_parts = []
                sentence_buffer = ""
                sentences_for_tts = []
                tts_queue = None
                tts_task = None
                tts_done_event = None
                
                if audio_on:
                    tts_queue = asyncio.Queue()
                    tts_done_event = asyncio.Event()
                    tts_task = asyncio.create_task(
                        _tts_pipeline(ws, tts_queue, voice_id, tts_done_event)
                    )
                
                routing_tag_found = False

                async for delta in ai.stream(
                    site_id=user_id,
                    system=system_prompt,
                    user=user_prompt,
                    images=image_attachments or None,
                ):
                    if delta.startswith("__USAGE__"):
                        try:
                            parts = delta[len("__USAGE__"):].split(",")
                            chat_input_tokens  += int(parts[0])
                            chat_output_tokens += int(parts[1])
                        except Exception:
                            pass
                        continue

                    full_text_parts.append(delta)
                    await ws.send_json({"type": "text_delta", "text": delta})

                    sentence_buffer += delta
                    while re.search(r'[.?!]\s', sentence_buffer):
                        match = re.search(r'[.?!]\s', sentence_buffer)
                        cut = match.end()
                        sentence = sentence_buffer[:cut].strip()
                        sentence_buffer = sentence_buffer[cut:]
                        if sentence and len(sentence) > 2:
                            sentences_for_tts.append(sentence)
                            if tts_queue and not routing_tag_found:
                                await tts_queue.put(sentence)

                # Handle remaining buffer
                remaining = sentence_buffer.strip()
                remaining = re.sub(r'\[?(NONE|INLINE|DIAGRAM|CODE|MATH|HTML)\]?\s*$', '', remaining, flags=re.IGNORECASE).strip()
                if remaining and len(remaining) > 2:
                    sentences_for_tts.append(remaining)
                    if tts_queue:
                        await tts_queue.put(remaining)

                if tts_done_event:
                    tts_done_event.set()

                raw_text = "".join(full_text_parts)
                print(f"==> RAW TAIL: {repr(raw_text[-80:])}", flush=True)
                bot_text, routing_decision = _extract_routing_tag(raw_text)
                print(f"==> ROUTING: {routing_decision}", flush=True)
                # --- CALENDAR TOOL DETECTION ---
                calendar_action = None
                calendar_payload = None

                cal_match = re.search(r'\[(CALENDAR_WRITE|CALENDAR_READ)\](\{.*?\})', raw_text, re.DOTALL)
                if cal_match:
                    calendar_action = cal_match.group(1)   # "CALENDAR_WRITE" or "CALENDAR_READ"
                    try:
                        calendar_payload = json.loads(cal_match.group(2))
                    except Exception as e:
                        print(f"==> Failed to parse calendar JSON: {e}")
                    
                    # Strip it from bot_text so user doesn't see the raw tag
                    bot_text = re.sub(r'\[(CALENDAR_WRITE|CALENDAR_READ)\]\{.*?\}', '', bot_text, flags=re.DOTALL).strip()

                bot_text = fix_markdown_formatting(bot_text)
                await ws.send_json({"type": "text_done", "text": bot_text})

                if not sentences_for_tts:
                    await ws.send_json({"type": "error", "message": "No response generated"})
                    await ws.send_json({"type": "done"})
                    if tts_task:
                        tts_task.cancel()
                    continue

                print(f"==> Routing decision: {routing_decision}", flush=True)

            except Exception as e:
                await ws.send_json({"type": "error", "message": f"AI failed: {str(e)}"})
                if tts_task:
                    tts_task.cancel()
                continue
            

            # ----------------------------------------------------------
            # WAIT FOR TTS TO FINISH
            # ----------------------------------------------------------
            if tts_task:
                try:
                    await asyncio.wait_for(tts_task, timeout=30)
                except asyncio.TimeoutError:
                    tts_task.cancel()
                    print("==> TTS task timed out, cancelled")
                except Exception as e:
                    print(f"==> TTS pipeline task error: {e}", flush=True)
            

            # Tell frontend no visual coming
            await ws.send_json({"type": "visual_aid_none"})


            # --- EXECUTE CALENDAR ACTION ---
            if calendar_action and calendar_payload:
                access_token = await get_valid_access_token(user_id, rag)
                if not access_token:
                    await ws.send_json({"type": "error", "message": "Microsoft not connected"})
                else:
                    try:
                        if calendar_action == "CALENDAR_WRITE":
                            result = await create_calendar_event(
                                access_token=access_token,
                                subject=calendar_payload["subject"],
                                start=datetime.fromisoformat(calendar_payload["start"]),
                                end=datetime.fromisoformat(calendar_payload["end"]),
                                body=calendar_payload.get("body", ""),
                                attendee_emails=calendar_payload.get("attendees", []),
                            )
                            await ws.send_json({"type": "calendar_done", "message": "Event booked!", "event": result})

                        elif calendar_action == "CALENDAR_READ":
                            events = await list_calendar_events(
                                access_token=access_token,
                                days_ahead=calendar_payload.get("days_ahead", 7),
                            )
                            await ws.send_json({"type": "calendar_events", "events": events})

                    except Exception as e:
                        print(f"==> Calendar action failed: {e}")
                        await ws.send_json({"type": "error", "message": f"Calendar action failed: {e}"}) #LATER WE NEED TO ALSO CALL THE AI AGAIN WITH ANOTHER RESPONSE TO THE FACT IT DIDNT WORK. //TWO CALLS

            # ----------------------------------------------------------
            # SAVE TO DB
            # ----------------------------------------------------------
            try:
                rag.add_message(chat_id=chat_id, role="assistant", content=bot_text)
                rag.update_last_message(chat_id=chat_id, last_message=bot_text)
            except Exception:
                pass        
            # ----------------------------------------------------------
            # Summary Builder
            # ----------------------------------------------------------
            try:
                char_count = 0
                cutoff = len(history)  # start assuming we use all of it

                for j in range(len(history) - 1, -1, -1):
                    char_count += len(history[j].get("content", ""))
                    if char_count > 800_000:
                        cutoff = j + 1  # everything from here forward is within budget
                        break


                trimmed_history = history[cutoff:]
                recent_for_summary = trimmed_history.copy()
                recent_for_summary.append({"role": "user", "content": user_text})
                recent_for_summary.append({"role": "assistant", "content": bot_text})
                await smgr.on_new_message(chat_id, recent_for_summary[-6:])
            except Exception as e:
                print(f"Summary update error: {e}")
            # ---------------------------------------------------------
            # MODEL 
            # ----------------------------------------------------------
            model = rag.get_model(user_id)

            # ----------------------------------------------------------
            # COST TRACKING
            # ----------------------------------------------------------
            try:
                billable_visual_text = ""

                cost = account_manager.processUsedCost(
                    # Call 1 — real tokens
                    input_tokens=chat_input_tokens+diagram_input_tokens,
                    output_tokens=chat_output_tokens+diagram_output_tokens,
                    # Everything else
                    SST_Length_seconds=len(audio_bytes) / 16000 if audio_bytes else 0,
                    webSearch=bool(web_search),
                    voice_on=bool(audio_on),
                    diagram_on=bool(False), #for agent #TEMP
                    pro_mode=bool(pro_mode),
                    image_count=len(image_attachments),
                    model=model
                )
                credits_used = cost / 0.15
                remaining = rag.deductCredits(user_id, credits_used)
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
        

# -----------------------------------------------------------------------
# FOR WEBSITE EMBEDDINGS
# -----------------------------------------------------------------------

@router.websocket("/embed_chat_ws")
async def embed_chat_ws(ws: WebSocket):
    print("HIT embed_chat_ws")
    t0 = time.time()
    await ws.accept()

    api_key = ws.query_params.get("api_key")
    if not api_key:
        await ws.close(code=4001, reason="Missing api_key")
        return

    initial_key_data = rag.getApiKey(api_key)
    if not initial_key_data or not initial_key_data.get("is_active"):
        await ws.close(code=4001, reason="Invalid or inactive API key")
        return

    owner_user_id = initial_key_data.get("owner_user_id")
    if not owner_user_id:
        await ws.close(code=4001, reason="API key has no owner")
        return

    import uuid
    embed_session_id = f"embed_{api_key}_{uuid.uuid4().hex[:12]}"
    MIN_CREDITS_PER_TURN = 0.05

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
            # RE-VALIDATE API KEY
            # ----------------------------------------------------------
            key_data = rag.getApiKey(api_key)
            if not key_data or not key_data.get("is_active"):
                await ws.send_json({"type": "error", "message": "API key deactivated", "code": "INVALID_KEY"})
                await ws.close()
                return

            # ----------------------------------------------------------
            # CONVERSATION LIMIT CHECK
            # ----------------------------------------------------------
            if key_data.get("conversations_used", 0) >= key_data.get("monthly_limit", 500):
                await ws.send_json({"type": "error", "message": "Monthly conversation limit reached", "code": "LIMIT_REACHED"})
                await ws.send_json({"type": "done"})
                continue

            # ----------------------------------------------------------
            # BUSINESS CREDIT CHECK
            # ----------------------------------------------------------
            try:
                current_credits = rag.getBusinessCredits(owner_user_id)
                if current_credits < MIN_CREDITS_PER_TURN:
                    await ws.send_json({"type": "error", "message": "This business has run out of credits. Please contact them to continue.", "code": "NO_CREDITS"})
                    await ws.send_json({"type": "done"})
                    continue
            except Exception as e:
                print(f"==> Credit check failed: {e}")
                await ws.send_json({"type": "error", "message": "Temporary billing error. Please try again shortly.", "code": "BILLING_ERROR"})
                await ws.send_json({"type": "done"})
                continue

            # ----------------------------------------------------------
            # DAILY COST CAP
            # ----------------------------------------------------------
            try:
                daily_cost = rag.getApiKeyDailyCost(api_key)
                daily_cap = key_data.get("daily_cost_cap", 10.00)
                if daily_cost >= daily_cap:
                    await ws.send_json({"type": "error", "message": "Daily usage cap reached. Try again tomorrow.", "code": "DAILY_CAP"})
                    await ws.send_json({"type": "done"})
                    continue
            except Exception:
                pass

            # ----------------------------------------------------------
            # EXTRACT PAYLOAD
            # ----------------------------------------------------------
            business_name = key_data.get("business_name") or "this business"
            business_description = key_data.get("business_description") or ""
            assistant_name = key_data.get("assistant_name") or "Assistant"
            version = key_data.get("assistant_version", "professional")
            model = rag.get_model(owner_user_id) or "gemini"

            user_text = _as_str(payload.get("message"))
            voice_id = _as_str(payload.get("voice_name"))
            audio_on = bool(payload.get("voice_on", True))
            raw_audio = payload.get("audio_bytes")
            session_id = _as_str(payload.get("session_id")) or f"embed_{api_key}_{uuid.uuid4().hex[:12]}"
            
            rag.get_or_create_embed_session(session_id, api_key, owner_user_id)

            if raw_audio and "," in raw_audio:
                raw_audio = raw_audio.split(",", 1)[1]
            audio_bytes = base64.b64decode(raw_audio) if raw_audio else None

            embed_input_tokens = 0
            embed_output_tokens = 0

            # ----------------------------------------------------------
            # CREDIT CHECK
            # ----------------------------------------------------------
            if not rag.hasEnoughCredits(owner_user_id):
                await ws.send_json({"type": "error", "message": "No credits remaining.", "code": "NO_CREDITS"})
                await ws.send_json({"type": "done"})
                continue

            # ----------------------------------------------------------
            # STT
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
            # VOICE ID FALLBACK
            # ----------------------------------------------------------
            if not voice_id:
                avatar = rag.getAvatarByName(key_data.get("avatar_name") or "Mia Sterling") or {}
                voice_id = avatar.get("voice") or "UgBBYS2sOqTuMpoF3BR0"

            # ----------------------------------------------------------
            # CONVERSATION HISTORY
            # ----------------------------------------------------------
            try:
                history = rag.get_recent_messages(user_id=owner_user_id, chat_id=session_id, limit=10)
            except Exception:
                history = []

            try:
                rag.add_message(chat_id=session_id, role="user", content=user_text)
            except Exception as e:
                print(f"==> Failed to save user message: {e}")

            # ----------------------------------------------------------
            # RAG LOOKUP
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
            tone_map = {
                "professional": "Be polite, professional, and clear. Sound like a well-trained support agent.",
                "friendly": "Be warm, casual, and approachable. Sound like a helpful friend who works at the company.",
                "concise": "Be extremely brief. One to two sentences max. No filler. Just the answer.",
            }
            tone = tone_map.get(version, tone_map["professional"])
            kb_section = rag_context if rag_context else "No information has been loaded yet."

            system_prompt = (
                f"You are {assistant_name}, a support assistant for {business_name}. "
                f"{business_description}\n\n"
                "RULES\n\n"
                f"1. ONLY use information from the knowledge base below to answer questions about {business_name}. "
                "If the answer is not there, say so directly and suggest the user contact the business.\n\n"
                "2. Never invent, guess, or fill in gaps. "
                "If you are not sure, say \"I don't have that information.\"\n\n"
                "3. Never make up contact details, policies, pricing, hours, or product features.\n\n"
                f"4. Stay on topic. You help with {business_name} only. "
                "Politely redirect off-topic questions.\n\n"
                "5. Keep responses under 80 words. Use short sentences. "
                "This will be spoken aloud, not read on screen.\n\n"
                f"6. {tone} Use \"I\" statements. Sound like a helpful person, not a corporate script.\n\n"
                "7. No markdown, no formatting, no lists, no headers, no bold, no asterisks. "
                "Plain conversational sentences only.\n\n"
                "8. When you don't know something, always offer a next step: "
                "\"You could reach out to them directly for that.\"\n\n"
                "9. Never say \"based on my training\", \"as an AI\", or \"I believe\". "
                "Just answer naturally or say you don't know.\n\n"
                "10. If someone asks who you are, say: "
                f"\"I'm {assistant_name}, a support assistant for {business_name}.\"\n\n"
                f"KNOWLEDGE BASE\n"
                f"Everything you know about {business_name} is below. "
                "If something is not here, you do not know it.\n\n"
                f"{kb_section}"
            )

            if audio_on:
                system_prompt += (
                    "\n\nLENGTH RULE: This response will be spoken aloud. "
                    "Keep it under 80 words. Lead with the core answer."
                )

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
            # GENERATE RESPONSE (streaming + per-sentence TTS)
            # ----------------------------------------------------------
            try:
                full_text_parts = []
                sentence_buffer = ""
                sentences_for_tts = []
                tts_queue = None
                tts_task = None
                tts_done_event = None

                if audio_on:
                    tts_queue = asyncio.Queue()
                    tts_done_event = asyncio.Event()
                    tts_task = asyncio.create_task(
                        _tts_pipeline(ws, tts_queue, voice_id, tts_done_event)
                    )

                print(f"==> embed stream starting at {time.time() - t0:.2f}s", flush=True)
                first_delta = True

                async for delta in ai.stream(
                    site_id=api_key,
                    system=system_prompt,
                    user=user_prompt,
                ):
                    if first_delta:
                        print(f"==> embed first token at {time.time() - t0:.2f}s", flush=True)
                        first_delta = False

                    # Capture real token usage from Anthropic
                    if delta.startswith("__USAGE__"):
                        try:
                            parts = delta[len("__USAGE__"):].strip().split(",")
                            embed_input_tokens += int(parts[0])
                            embed_output_tokens += int(parts[1])
                        except Exception:
                            pass
                        continue

                    full_text_parts.append(delta)
                    await ws.send_json({"type": "text_delta", "text": delta})

                    sentence_buffer += delta
                    while re.search(r'[.?!]\s', sentence_buffer):
                        match = re.search(r'[.?!]\s', sentence_buffer)
                        cut = match.end()
                        sentence = sentence_buffer[:cut].strip()
                        sentence_buffer = sentence_buffer[cut:]

                        # Filter garbage before sending to TTS
                        ROUTING_TAGS = {'none', 'inline', 'diagram', 'code', 'math', 'html'}
                        if sentence and len(sentence) > 2 and sentence.strip().lower() not in ROUTING_TAGS:
                            cleaned_sentence = re.sub(
                                r'\[?(NONE|INLINE|DIAGRAM|CODE|MATH|HTML)\]?\s*$', '',
                                sentence, flags=re.IGNORECASE
                            ).strip()
                            if cleaned_sentence and len(cleaned_sentence) > 2:
                                sentences_for_tts.append(cleaned_sentence)
                                if tts_queue:
                                    await tts_queue.put(cleaned_sentence)

                # Handle remaining buffer
                remaining = sentence_buffer.strip()
                remaining = re.sub(
                    r'\[?(NONE|INLINE|DIAGRAM|CODE|MATH|HTML)\]?\s*$', '',
                    remaining, flags=re.IGNORECASE
                ).strip()
                if remaining and len(remaining) > 2:
                    sentences_for_tts.append(remaining)
                    if tts_queue:
                        await tts_queue.put(remaining)

                if tts_done_event:
                    tts_done_event.set()

                bot_text = "".join(full_text_parts).strip()
                bot_text = re.sub(r'```[a-z]*\n?.*?```', '', bot_text, flags=re.DOTALL).strip()
                bot_text = re.sub(r'\n\s*\n', '\n\n', bot_text)

                await ws.send_json({"type": "text_done", "text": bot_text})
                print(f"==> embed stream done at {time.time() - t0:.2f}s", flush=True)

                if not sentences_for_tts:
                    await ws.send_json({"type": "error", "message": "No response generated"})
                    await ws.send_json({"type": "done"})
                    if tts_task:
                        tts_task.cancel()
                    continue

            except Exception as e:
                await ws.send_json({"type": "error", "message": "The assistant is unavailable right now. Please try again shortly."})
                print(f"==> embed AI failed: {e}")
                traceback.print_exc()
                if tts_task:
                    tts_task.cancel()
                await ws.send_json({"type": "done"})
                continue

            # ----------------------------------------------------------
            # WAIT FOR TTS
            # ----------------------------------------------------------
            if tts_task:
                try:
                    await asyncio.wait_for(tts_task, timeout=30)
                except asyncio.TimeoutError:
                    tts_task.cancel()
                    print("==> embed TTS task timed out, cancelled")
                except Exception as e:
                    print(f"==> embed TTS pipeline error: {e}", flush=True)

            # ----------------------------------------------------------
            # SAVE ASSISTANT RESPONSE
            # ----------------------------------------------------------
            try:
                rag.add_message(chat_id=session_id, role="assistant", content=bot_text)
                rag.update_last_message(chat_id=session_id, last_message=bot_text)
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
            # COST TRACKING + CREDIT DEDUCTION
            # ----------------------------------------------------------
            try:
                cost_aud = account_manager.processUsedCost(
                    input_tokens=embed_input_tokens,
                    output_tokens=embed_output_tokens,
                    outputText=bot_text,
                    SST_Length_seconds=len(audio_bytes) / 32000 if audio_bytes else 0,
                    webSearch=False,
                    voice_on=bool(audio_on),
                    diagram_on=False,
                    model=model,
                )

                try:
                    rag.addApiKeyCost(api_key, cost_aud)
                except Exception as e:
                    print(f"==> Failed to add api_key cost: {e}")

                try:
                    new_balance = rag.deductBusinessCredits(owner_user_id, cost_aud)
                    print(f"==> Embed cost: ${cost_aud:.4f} AUD | tokens in={embed_input_tokens} out={embed_output_tokens} | balance: ${new_balance:.4f}", flush=True)

                    if new_balance < 0.50:
                        await ws.send_json({"type": "credits_low", "balance": new_balance, "message": "Credits running low"})
                except Exception as e:
                    print(f"==> Failed to deduct business credits: {e}")

            except Exception as e:
                print(f"==> Embed cost tracking failed: {e}", flush=True)

            # ----------------------------------------------------------
            # DONE
            # ----------------------------------------------------------
            try:
                await ws.send_json({"type": "done"})
                print("==> embed DONE sent", flush=True)
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