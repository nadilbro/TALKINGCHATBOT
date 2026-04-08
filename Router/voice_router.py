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

@router.post("/chat_diagram_init")
async def chat_diagram_init(init_details: SessionInit, user=Depends(verify_token)):
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

    return {"visuals": visuals}
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
            raw_file = payload.get("file_bytes")
            file_name = payload.get("file_name")
            if raw_audio and "," in raw_audio:
                raw_audio = raw_audio.split(",", 1)[1]
            audio_bytes = base64.b64decode(raw_audio) if raw_audio else None

            # ----------------------------------------------------------
            # CREDIT CHECK
            # ----------------------------------------------------------
            if not rag.hasEnoughCredits(user_id):
                await ws.send_json({
                    "type": "error",
                    "message": "You have no credits remaining. Please top up to continue chatting.",
                    "code": "NO_CREDITS"
                })
                await ws.send_json({"type": "done"})
                continue
 
            # ----------------------------------------------------------
            # SPEECH TO TEXT
            # ----------------------------------------------------------
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
            # CHECK FILE INPUT
            # ----------------------------------------------------------
            file_context = ""
            if raw_file and file_name:
                if "," in raw_file:
                    raw_file = raw_file.split(",", 1)[1]
                file_bytes_decoded = base64.b64decode(raw_file)
                try:
                    file_context = await fileE.extract_text(file_bytes_decoded, file_name)
                    print(f"==> File extracted: {file_name}, length={len(file_context)}")
                except Exception as e:
                    print(f"==> File extraction failed: {e}")
                    traceback.print_exc()
                    await ws.send_json({"type": "error", "message": f"File read failed: {e}"})
            else:
                print(f"==> No file received. raw_file={bool(raw_file)}, file_name={file_name}")
            # ----------------------------------------------------------
            # BUILD BASE PROMPT
            # ----------------------------------------------------------
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
            # CHECK IF VISUAL AIDS ARE ENABLED
            # ----------------------------------------------------------
            try:
                diagrams_enabled = rag.get_diagram_usage(user_id)
            except Exception:
                diagrams_enabled = False
 
            # ----------------------------------------------------------
            # VISUAL AID GENERATION (diagram OR code, runs FIRST)
            # ----------------------------------------------------------
            async def _generate_visual_aid():
                """Decide whether to generate a diagram, code, or nothing.
                Returns one of:
                    {"type": "diagram", "svg": "<svg>...</svg>"}
                    {"type": "code", "language": "python", "code": "..."}
                    None
                """
                if not diagrams_enabled:
                    print("Visual aids disabled for this user")
                    return None
                try:
                    print("Generating visual aid (pre-chat)...")
                    raw = await ai.get_diagram(site_id=user_id, user=user_text)
                    print(f"==> Raw visual aid response: {raw[:500]!r}")
 
                    if not raw:
                        return None
 
                    stripped = raw.strip()
                    if not stripped:
                        return None
 
                    first_word = stripped.split(None, 1)[0].upper()
 
                    # NONE — nothing useful to generate
                    if first_word == "NONE":
                        print("Visual aid model returned NONE")
                        return None
 
                    # DIAGRAM — extract the SVG block
                    if first_word == "DIAGRAM":
                        match = re.search(r'<svg.*?</svg>', raw, re.DOTALL | re.IGNORECASE)
                        if match:
                            return {"type": "diagram", "svg": match.group(0)}
                        if '<svg' in raw.lower() and '</svg>' not in raw.lower():
                            print(f"==> Diagram response appears truncated ({len(raw)} chars)")
                            return None
                        print("DIAGRAM marker found but no <svg> block in response")
                        return None
 
                    # CODE — parse language and body
                    if first_word == "CODE":
                        lines = stripped.split("\n", 2)
                        if len(lines) < 3:
                            print(f"==> CODE response malformed (need 3+ lines, got {len(lines)})")
                            return None
 
                        language = lines[1].strip().lower()
                        code_body = lines[2]
 
                        if not language or not re.match(r'^[a-z0-9+#\-]+$', language):
                            print(f"==> CODE response had invalid language: {language!r}")
                            return None
 
                        # Strip any accidental markdown fences
                        code_body = re.sub(r'^```[\w]*\n?', '', code_body)
                        code_body = re.sub(r'\n?```$', '', code_body)
                        code_body = code_body.strip("\n")
 
                        if not code_body:
                            print("==> CODE response had empty body")
                            return None
 
                        return {"type": "code", "language": language, "code": code_body}
 
                    # Fallback — unexpected format, try to salvage an SVG
                    match = re.search(r'<svg.*?</svg>', raw, re.DOTALL | re.IGNORECASE)
                    if match:
                        print("Unexpected format but SVG found, treating as diagram")
                        return {"type": "diagram", "svg": match.group(0)}
 
                    print(f"Visual aid response had unknown first word: {first_word!r}")
                    return None
 
                except Exception as e:
                    print(f"==> Visual aid generation failed: {e}", flush=True)
                    return None
 
            # Optional UX hint to the frontend
            if diagrams_enabled:
                try:
                    await ws.send_json({"type": "visual_aid_pending"})
                except Exception:
                    pass
 
            visual_aid = await _generate_visual_aid()
 
            # Send the visual aid immediately so the frontend can render it
            # while the chat response is still streaming
            if visual_aid is None:
                try:
                    await ws.send_json({"type": "visual_aid_none"})
                except Exception:
                    pass
            elif visual_aid["type"] == "diagram":
                await ws.send_json({"type": "diagram", "svg": visual_aid["svg"]})
            elif visual_aid["type"] == "code":
                await ws.send_json({
                    "type": "code",
                    "language": visual_aid["language"],
                    "code": visual_aid["code"],
                })

            if visual_aid:
                try:
                    rag.save_visual(
                        session_id=chat_id,
                        visual_type=visual_aid["type"],
                        content=visual_aid["svg"] if visual_aid["type"] == "diagram" else visual_aid["code"],
                        language=visual_aid.get("language"),
                    )
                except Exception as e:
                    print(f"==> Failed to save visual: {e}")
 
            # ----------------------------------------------------------
            # BUILD GROUNDING CONTEXT FOR THE CHARACTER
            # ----------------------------------------------------------
            visual_aid_summary = ""
 
            if visual_aid and visual_aid["type"] == "diagram":
                svg_text = visual_aid["svg"]
                labels = re.findall(r'<text[^>]*>(.*?)</text>', svg_text, re.DOTALL | re.IGNORECASE)
                labels = [re.sub(r'\s+', ' ', lbl).strip() for lbl in labels if lbl.strip()]
                if labels:
                    visual_aid_summary = (
                        "A diagram has been shown to the user alongside your response. "
                        "It contains the following labels and elements: "
                        + ", ".join(labels) + ". "
                        "Speak naturally about the topic. Your explanation should be "
                        "consistent with these diagram elements, but do not announce "
                        "the diagram, do not say 'as you can see', and do not describe "
                        "the diagram in words. Just explain the concept."
                    )
 
            elif visual_aid and visual_aid["type"] == "code":
                language = visual_aid["language"]
                code_preview = visual_aid["code"]
                if len(code_preview) > 1500:
                    code_preview = code_preview[:1500] + "\n... (truncated)"
                visual_aid_summary = (
                    f"A code snippet in {language} has been shown to the user alongside "
                    f"your response. The code is:\n\n{code_preview}\n\n"
                    "Speak naturally about what the code does and how it works, as if "
                    "you were explaining it to a friend out loud. Do not read the code "
                    "line by line. Do not announce that code has been shown. Do not say "
                    "'here is the code' or 'as you can see'. Just walk them through the "
                    "approach conversationally. Keep it concise — the user can read the "
                    "code themselves, your job is to give them the intuition behind it."
                )
 
            if visual_aid_summary:
                # Visual aid was generated — character grounds their response in it
                system_prompt = f"{system_prompt}\n\n{visual_aid_summary}"
            
            elif visual_aid is None and diagrams_enabled:
                # Visual aids ARE enabled, but the router decided nothing was needed
                # for this specific question. Character just responds normally.
                system_prompt = (
                    f"{system_prompt}\n\n"
                    "No visual aid was generated for this question because a diagram or "
                    "code snippet would not meaningfully help. Respond conversationally "
                    "as you normally would."
                )
            
            else:
                # Visual aids are DISABLED by the user. The character should still answer
                # helpfully, and if the question would have benefited from code or a
                # diagram, should mention that enabling visual aids would let them show
                # it properly.
                system_prompt = (
                    f"{system_prompt}\n\n"
                    "IMPORTANT: The user has visual aids turned OFF. This means no "
                    "diagrams and no code blocks can be shown to them right now. You "
                    "must still answer their question fully and helpfully in your "
                    "spoken voice. Do not refuse to answer just because you cannot show "
                    "a visual.\n\n"
                    "If the question is about code or programming, explain the concept "
                    "and the approach in plain conversational words. Walk through what "
                    "the code would do step by step as if you were describing it out "
                    "loud to a friend. Do not output code blocks, markdown, or syntax — "
                    "just explain the logic and approach verbally. At the end of your "
                    "answer, briefly mention that if they want to see the actual code "
                    "formatted nicely, they can enable the visual aids button in the "
                    "chat interface.\n\n"
                    "If the question is about a concept that would normally be easier "
                    "with a diagram, explain it clearly in words and mention at the end "
                    "that enabling the visual aids button would let you show them a "
                    "diagram too.\n\n"
                    "If the question is casual or doesn't need a visual at all, just "
                    "answer normally without mentioning visual aids — do not bring it "
                    "up for every response, only when it would genuinely have helped."
                )

            #ADDING FILE CONTEXT
            if file_context:
                print("File attached")
                print(file_context)
                system_prompt = f"{system_prompt}\n\nUser has attached a file as follows: ({file_name}):\n{file_context}"
            else:
                print("File Not attached")
                system_prompt = f"{system_prompt}\n\nUSER HAS NOT UPLOADED ANY EXTRA FILES"

            # ----------------------------------------------------------
            # GENERATE CHAT RESPONSE
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
 
                # Strip any stray code fences Flash might sneak in
                bot_text = re.sub(r'```[a-z]*\n?.*?```', '', bot_text, flags=re.DOTALL).strip()
                bot_text = re.sub(r'\n\s*\n', '\n\n', bot_text)
 
                # Rebuild sentences from cleaned text
                sentences = [
                    s.strip()
                    for s in re.split(r'(?<=[.?!])\s+', bot_text)
                    if len(s.strip()) > 2
                ]
 
                if not sentences:
                    await ws.send_json({"type": "error", "message": "No response generated"})
                    await ws.send_json({"type": "done"})
                    continue
 
                aid_kind = "none"
                if visual_aid:
                    aid_kind = visual_aid["type"]
                print(f"==> {len(sentences)} sentences | visual_aid={aid_kind}", flush=True)
 
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"AI failed: {str(e)}"})
                continue
 
            # Send the spoken text
            await ws.send_json({"type": "text", "text": bot_text})
 
            # Persist assistant message
            try:
                rag.add_message(chat_id=chat_id, role="assistant", content=bot_text)
                rag.update_last_message(chat_id=chat_id, last_message=bot_text)
            except Exception:
                pass
 
            # Rolling summary
            try:
                smgr = get_summary_manager()
                recent_for_summary = history[-6:] if len(history) > 6 else list(history)
                recent_for_summary.append({"role": "user", "content": user_text})
                recent_for_summary.append({"role": "assistant", "content": bot_text})
                await smgr.on_new_message(chat_id, recent_for_summary[-6:])
            except Exception as e:
                print(f"Summary update error: {e}")
 
            # ----------------------------------------------------------
            # TEXT TO SPEECH
            # ----------------------------------------------------------
            if audio_on:
                try:
                    print(f"==> Firing {len(sentences)} ElevenLabs tasks in parallel", flush=True)
                    all_audio, all_visemes, total_duration_seconds = await _run_tts(sentences, voice_id)
 
                    if not all_audio:
                        await ws.send_json({"type": "error", "message": "TTS produced no audio"})
                    else:
                        await _send_audio(ws, all_audio, all_visemes)
 
                except Exception as e:
                    print(f"==> TTS error: {e}", flush=True)
                    traceback.print_exc()
                    await ws.send_json({"type": "error", "message": f"TTS failed: {str(e)}"})
 
            # ----------------------------------------------------------
            # COST TRACKING
            # ----------------------------------------------------------
            try:
                # Build the "diagram text" billable string from whichever
                # visual aid (if any) was generated
                billable_visual_text = ""
                if visual_aid and visual_aid["type"] == "diagram":
                    billable_visual_text = visual_aid["svg"]
                elif visual_aid and visual_aid["type"] == "code":
                    billable_visual_text = visual_aid["code"]
                
                print(f"==> bot_text length={len(bot_text)}, preview={bot_text[:200]!r}")
                print(f"==> visual_aid length={len(billable_visual_text)}")
                print(f"For testing sake: input text: {user_prompt}")
 
                cost = account_manager.processUsedCost(
                    outputText=bot_text,
                    outputDiagramText=billable_visual_text,
                    inputText=user_prompt,
                    SST_Length_seconds=len(audio_bytes) / 16000 if audio_bytes else 0,
                    webSearch=bool(web_search),
                    voice_on=bool(audio_on),
                    diagram_on=bool(visual_aid),
                )
                credits_used = cost / 0.15
                remaining = rag.deductCredits(user_id, credits_used)
                print(f"==> Cost: ${cost:.4f} | Deducted {credits_used:.4f} credits. Remaining: {remaining}", flush=True)
            except Exception as e:
                print(f"==> Cost tracking failed: {e}", flush=True)
 
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

@router.websocket("/embed_chat_ws")
async def embed_chat_ws(ws: WebSocket):
    """
    WebSocket endpoint for embedded widget.
    Auth via API key in query param. Billing per API key.
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

    # Give each websocket session a conversation_id so we can thread history.
    # If your rag layer doesn't support per-session embed conversations yet,
    # you'll need to add a simple `embed_sessions` table: (session_id, api_key, created_at).
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

    print(f"==> embed WS opened for api_key={api_key}, session={embed_session_id}")

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

            # Re-fetch key_data every message so dashboard toggles apply mid-session
            key_data = rag.getApiKey(api_key)
            if not key_data or not key_data.get("is_active"):
                await ws.send_json({"type": "error", "message": "API key deactivated", "code": "INVALID_KEY"})
                await ws.close()
                return

            if key_data.get("conversations_used", 0) >= key_data.get("monthly_limit", 500):
                await ws.send_json({"type": "error", "message": "Monthly conversation limit reached", "code": "LIMIT_REACHED"})
                await ws.close()
                return

            # Per-API-key daily cost cap (implement rag.getApiKeyDailyCost / rag.getApiKeyDailyCap)
            try:
                daily_cost = rag.getApiKeyDailyCost(api_key)
                daily_cap = key_data.get("daily_cost_cap", 10.00)  # $10 default
                if daily_cost >= daily_cap:
                    await ws.send_json({
                        "type": "error",
                        "message": "Daily usage cap reached. Try again tomorrow.",
                        "code": "DAILY_CAP"
                    })
                    continue
            except Exception:
                pass  # if cap check fails, fail open (log it)

            business_name = key_data.get("business_name") or ""
            business_description = key_data.get("business_description") or ""
            avatar_name = key_data.get("avatar_name") or "Mia Sterling"
            personality_on = bool(key_data.get("personality_on"))

            user_text = _as_str(payload.get("message"))
            voice_id = _as_str(payload.get("voice_name"))
            audio_on = bool(payload.get("voice_on", True))
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

            # Look up avatar once
            avatar = rag.getAvatarByName(avatar_name) or {}

            # Resolve voice
            if not voice_id:
                voice_id = avatar.get("voice") or "UgBBYS2sOqTuMpoF3BR0"

            # Load conversation history for this embed session
            try:
                history = rag.get_recent_messages_by_session(
                    session_id=embed_session_id,
                    limit=10,
                )
            except Exception:
                history = []

            # Save the incoming user message
            try:
                rag.add_embed_message(
                    session_id=embed_session_id,
                    api_key=api_key,
                    role="user",
                    content=user_text,
                )
            except Exception as e:
                print(f"==> Failed to save user message: {e}")

            # RAG lookup (business knowledge base)
            rag_context = ""
            try:
                embedding = await rag.embedText(user_text)
                chunks = rag.searchDocumentChunks(api_key=api_key, embedding=embedding, limit=3)
                if chunks and chunks[0].get("similarity", 0) >= 0.3:
                    rag_context = "\n".join(f"- {c['content']}" for c in chunks)
            except Exception as e:
                print(f"==> RAG failed: {e}")

            # Build system prompt — always prepend formatting rule
            if personality_on:
                avatar_prompt = avatar.get("prompt") or ""
                fallback_prompt = key_data.get("system_prompt") or ""
                persona = avatar_prompt or fallback_prompt
                system_prompt = f"{FORMATTING_RULE}\n\n{persona}"
            else:
                system_prompt = (
                    f"{FORMATTING_RULE}\n\n"
                    f"You are a helpful support agent for {business_name}. "
                    f"{business_description} "
                    f"Answer the user's questions accurately and helpfully. "
                    f"Do not make up information you do not have. "
                    f"If you don't know the answer, say so honestly. "
                    f"Keep responses conversational and under 100 words unless the user asks for detail."
                )

            if rag_context:
                system_prompt = f"{system_prompt}\n\nRelevant information from the business knowledge base:\n{rag_context}"

            # Build user prompt with history
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

            # Stream Gemini
            try:
                sentences = []
                sentence_buffer = ""
                async for delta in ai.stream(site_id=api_key, system=system_prompt, user=user_prompt):
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
                continue

            # Send text to client
            await ws.send_json({"type": "text", "text": bot_text})

            # Save assistant response
            try:
                rag.add_embed_message(
                    session_id=embed_session_id,
                    api_key=api_key,
                    role="assistant",
                    content=bot_text,
                )
            except Exception as e:
                print(f"==> Failed to save assistant message: {e}")

            # Increment conversation count
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
                    await ws.send_json({"type": "error", "message": "Voice playback failed."})

            # Cost tracking — ALWAYS runs, regardless of audio_on
            try:
                cost_aud = account_manager.processUsedCost(
                    outputText=bot_text,
                    outputDiagramText="",
                    inputText=user_prompt,
                    SST_Length_seconds=len(audio_bytes) / 16000 if audio_bytes else 0,
                    webSearch=False,
                    voice_on=bool(audio_on),
                    diagram_on=False,
                )
                rag.addApiKeyCost(api_key, cost_aud)
                print(f"==> Embed cost: ${cost_aud:.4f} AUD for api_key={api_key}", flush=True)
            except Exception as e:
                print(f"==> Embed cost tracking failed: {e}", flush=True)

            await ws.send_json({"type": "done"})

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