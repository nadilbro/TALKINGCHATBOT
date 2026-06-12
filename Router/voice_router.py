import re
import asyncio
import traceback
import base64
import time
import uuid
import json
from html import unescape
from typing import Any, Optional
from datetime import datetime, timezone

from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from Providers.firebase_auth import verify_token, verify_ws_token
from Providers.ai_provider import AIProvider
from Providers.voice_chat import VoiceChatSystem
from SQL.SQLManager import VectorRAGService
from Providers.APIContracts import SessionInit, DiagramInit
from Providers.web_search import TavilyProvider
from Providers.summary_generator import RollingSummaryManager
from Providers.STT import DeepgramProvider
from Providers.Account_Manager import AccountManager
from Providers.file_extractor import FileExtractor
from Providers.Integrations.microsoft_auth import (
    get_valid_access_token as ms_get_token,
    create_calendar_event as ms_create_event,
    list_calendar_events as ms_list_events,
    delete_calendar_event as ms_delete_event,
)
from Providers.Integrations.google_auth import (
    get_valid_access_token as google_get_token,
    create_calendar_event as google_create_event,
    list_calendar_events as google_list_events,
    delete_calendar_event as google_delete_event,
)

router = APIRouter(prefix="/system", tags=["chat"])

rag = VectorRAGService()
ai = AIProvider(rag)
account_manager = AccountManager(rag)
fileE = FileExtractor(ai)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MINUTES_PER_CREDIT = 7          # kept for external imports
AUD_PER_CREDIT = 0.15           # main-app credit conversion
STT_BYTES_PER_SECOND = 32_000   # 16 kHz 16-bit mono PCM
AUDIO_CHUNK_BYTES = 32_000      # ws binary frame size for outgoing audio
TTS_TIMEOUT_S = 30
MAX_FILES = 5
MAX_TOTAL_UPLOAD_BYTES = 40 * 1024 * 1024
MIN_CREDITS_PER_TURN = 0.05     # embed
DEFAULT_VOICE_ID = "UgBBYS2sOqTuMpoF3BR0"

# Time-to-first-audio: if the model's FIRST sentence runs long, flush its
# first clause (at a comma) to TTS early so the avatar starts speaking
# sooner. Only ever applies before the first TTS chunk of a turn.
EARLY_FIRST_CLAUSE_FLUSH = True
EARLY_FLUSH_MIN_CHARS = 90      # buffer length before considering a flush
EARLY_FLUSH_MIN_CLAUSE = 40     # don't flush clauses shorter than this

# ---------------------------------------------------------------------------
# Patterns (compiled once)
# ---------------------------------------------------------------------------
ACTION_TAG_PATTERN = re.compile(
    r'\[?(CALENDAR_WRITE|CALENDAR_READ|CALENDAR_DELETE|'
    r'GOOGLE_CALENDAR_WRITE|GOOGLE_CALENDAR_READ|GOOGLE_CALENDAR_DELETE|'
    r'WEB_SEARCH)\]?\s*\{[^}]*\}',
    re.IGNORECASE,
)

ROUTING_TAG_PATTERN = re.compile(
    r'\[?(NONE|INLINE|DIAGRAM|CODE|MATH|HTML|LINK)\]?\s*$',
    re.IGNORECASE,
)

ROUTING_TAG_END_RE = re.compile(
    r'\[(NONE|INLINE|DIAGRAM|CODE|MATH|HTML|LINK)\]\s*$', re.IGNORECASE
)

CALENDAR_TAG_RE = re.compile(
    r'\[(CALENDAR_WRITE|CALENDAR_READ|CALENDAR_DELETE|'
    r'GOOGLE_CALENDAR_WRITE|GOOGLE_CALENDAR_READ|GOOGLE_CALENDAR_DELETE)\](\{[^}]*\})'
)

WEB_SEARCH_RE = re.compile(r'\[WEB_SEARCH\]\{"query":\s*"([^"]+)"\}')
LINK_TAG_RE = re.compile(r'\[LINK\]\s*(\{[^}]*\})')
SENTENCE_END_RE = re.compile(r'[.?!]\s')
CODE_FENCE_RE = re.compile(r'```[a-z]*\n?.*?```', re.DOTALL)

_ROUTING_TAG_WORDS = {'none', 'inline', 'diagram', 'code', 'math', 'html', 'link'}

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_tts = None
_stt = None
_tav = None
_summary_mgr = None


def get_tts():
    global _tts
    if _tts is None:
        _tts = VoiceChatSystem()
    return _tts


def get_stt():
    global _stt
    if _stt is None:
        _stt = DeepgramProvider()
    return _stt


def get_web_search():
    global _tav
    if _tav is None:
        _tav = TavilyProvider()
    return _tav


def get_summary_manager():
    global _summary_mgr
    if _summary_mgr is None:
        _summary_mgr = RollingSummaryManager(
            gemini_provider=ai._providers["gemini_flash"],
            rag=rag,
        )
    return _summary_mgr


# ---------------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------------
def _as_str(x: Any) -> str:
    return (str(x) if x is not None else "").strip()


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


async def _drain(task):
    """Await a prefetch task on early-exit paths so threads finish cleanly."""
    if task is None:
        return None
    try:
        return await task
    except Exception as e:
        print(f"==> Prefetch drain error: {e}")
        return None


def strip_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'```[\w]*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def fix_markdown_formatting(text: str) -> str:
    text = re.sub(r'(?<!\n)\n(#{1,6}\s)', r'\n\n\1', text)
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
    return text


def html_to_plain_text(html_text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", html_text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------
def _extract_routing_tag(text: str) -> tuple:
    match = ROUTING_TAG_END_RE.search(text)
    if match:
        tag = match.group(1).upper()
        return text[:match.start()].rstrip(), tag
    lines = text.strip().rsplit('\n', 1)
    if len(lines) == 2:
        last_line = lines[1].strip().upper()
        if last_line.strip('[]') in {w.upper() for w in _ROUTING_TAG_WORDS}:
            return lines[0].rstrip(), last_line.strip('[]')
    return text, "NONE"


def _detect_link(raw_text: str):
    return LINK_TAG_RE.search(raw_text)


def _detect_calendar_action(raw_text: str) -> tuple:
    cal_match = CALENDAR_TAG_RE.search(raw_text)
    if not cal_match:
        return None, None
    action = cal_match.group(1)
    try:
        return action, json.loads(cal_match.group(2))
    except Exception as e:
        print(f"==> Failed to parse calendar JSON: {e} | raw: {repr(cal_match.group(2))}")
        return None, None


def _detect_web_search(raw_text: str):
    return WEB_SEARCH_RE.search(raw_text)


def _clean_bot_text(raw_text: str) -> str:
    """Strips ALL action tags, routing tags, and artifacts before sending to frontend."""
    text = ACTION_TAG_PATTERN.sub('', raw_text)
    text, _ = _extract_routing_tag(text)
    text = CODE_FENCE_RE.sub('', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def _tts_clean(sentence: str) -> str:
    """Cleans a sentence before sending to ElevenLabs. Strips all tags."""
    cleaned = strip_markdown(sentence)
    cleaned = ACTION_TAG_PATTERN.sub('', cleaned)
    cleaned = ROUTING_TAG_PATTERN.sub('', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned or len(cleaned) < 3:
        return ''
    if cleaned.lower() in _ROUTING_TAG_WORDS:
        return ''
    return cleaned


# ---------------------------------------------------------------------------
# TTS pipeline
# ---------------------------------------------------------------------------
async def _tts_pipeline(ws, sentences_queue: asyncio.Queue, voice_id: str, done_event: asyncio.Event):
    tts_instance = get_tts()
    cumulative_offset_ms = 0
    first_chunk = True

    while True:
        try:
            sentence = await asyncio.wait_for(sentences_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            if done_event.is_set() and sentences_queue.empty():
                break
            continue

        if sentence is None:
            break

        try:
            cleaned = _tts_clean(sentence)
            if not cleaned:
                continue

            audio_bytes_chunk, visemes, duration = await tts_instance.synthesize_sentence(
                cleaned, voice_id
            )
            if not audio_bytes_chunk:
                continue

            chunk_visemes = [
                {"t_ms": _as_int(v.get("t_ms"), 0) + cumulative_offset_ms,
                 "viseme_id": _as_int(v.get("viseme_id"), 0)}
                for v in visemes
            ]

            if first_chunk:
                await ws.send_json({
                    "type": "audio_begin",
                    "format": "mp3",
                    "sample_rate_hz": 44100,
                    "channels": 1,
                    "visemes": chunk_visemes,
                })
                first_chunk = False
            elif chunk_visemes:
                await ws.send_json({"type": "viseme_update", "visemes": chunk_visemes})

            for i in range(0, len(audio_bytes_chunk), AUDIO_CHUNK_BYTES):
                await ws.send_bytes(audio_bytes_chunk[i:i + AUDIO_CHUNK_BYTES])

            cumulative_offset_ms += int(duration * 1000)

        except Exception as e:
            print(f"==> TTS pipeline error: {e}", flush=True)
            traceback.print_exc()

    if not first_chunk:
        try:
            await ws.send_json({"type": "audio_end"})
        except Exception:
            pass


def _start_tts(ws, voice_id: str, audio_on: bool):
    """Returns (queue, task, done_event) — all None when audio is off."""
    if not audio_on:
        return None, None, None
    tts_queue = asyncio.Queue()
    done_event = asyncio.Event()
    task = asyncio.create_task(_tts_pipeline(ws, tts_queue, voice_id, done_event))
    return tts_queue, task, done_event


async def _await_tts(tts_task, label: str = ""):
    if not tts_task:
        return
    try:
        await asyncio.wait_for(tts_task, timeout=TTS_TIMEOUT_S)
    except asyncio.TimeoutError:
        tts_task.cancel()
        print(f"==> TTS timed out {label}")
    except Exception as e:
        print(f"==> TTS error {label}: {e}", flush=True)


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------
async def _transcribe(ws, audio_bytes: bytes) -> Optional[str]:
    """Threadpooled STT (Deepgram client is blocking). None on failure."""
    try:
        text = await run_in_threadpool(get_stt().get_transcript, audio_bytes)
    except Exception as e:
        traceback.print_exc()
        await ws.send_json({"type": "error", "message": f"Transcription failed: {e}"})
        return None
    if not text:
        await ws.send_json({"type": "error", "message": "Could not understand audio. Try again."})
        return None
    await ws.send_json({"type": "transcript", "text": text})
    return text


# ---------------------------------------------------------------------------
# Per-turn state prefetch — ONE threadpool hop for all sync DB reads,
# started BEFORE STT so the round-trips overlap transcription (300-800ms)
# instead of running serially after it.
# ---------------------------------------------------------------------------
def _prefetch_main_state(user_id: str, chat_id: str, need_avatar: bool, smgr) -> dict:
    state = {
        "has_credits": rag.hasEnoughCredits(user_id),
        "history": rag.get_recent_messages(user_id=user_id, chat_id=chat_id, limit=20),
        "summary": smgr.build_context(chat_id),
        "integrator": rag.checkIntegrations(user_id),
        "default_integration": rag.getDefaultIntegration(user_id),
        "model": rag.get_model(user_id),
        "avatar": None,
    }
    if need_avatar:
        try:
            state["avatar"] = rag.get_avatar(user_id, chat_id)
        except Exception:
            pass
    return state


def _prefetch_embed_turn(api_key: str, owner_user_id: str) -> dict:
    return {
        "key_data": rag.getApiKey(api_key),
        "business_credits": rag.getBusinessCredits(owner_user_id),
        "has_credits": rag.hasEnoughCredits(owner_user_id),
    }


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
async def _process_files(ws, files_payload: list, user_id: str) -> tuple:
    file_context = ""
    image_attachments = []
    file_text_parts = []
    total_upload_bytes = 0

    for idx, f in enumerate(files_payload):
        raw_file = f.get("file_bytes")
        file_name = f.get("file_name")
        if not raw_file or not file_name:
            continue
        if "," in raw_file:
            raw_file = raw_file.split(",", 1)[1]
        try:
            file_bytes_decoded = base64.b64decode(raw_file)
        except Exception:
            await ws.send_json({"type": "error", "message": f"Could not decode {file_name}."})
            continue

        total_upload_bytes += len(file_bytes_decoded)
        if total_upload_bytes > MAX_TOTAL_UPLOAD_BYTES:
            await ws.send_json({
                "type": "error",
                "message": f"Total upload exceeds {MAX_TOTAL_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
                "code": "UPLOAD_TOO_LARGE",
            })
            return "", [], True

        try:
            ext = file_name.lower().split(".")[-1]
            if ext in ("png", "jpg", "jpeg", "webp"):
                mime_map = {"png": "image/png", "jpg": "image/jpeg",
                            "jpeg": "image/jpeg", "webp": "image/webp"}
                image_attachments.append({"mime_type": mime_map[ext], "data": file_bytes_decoded})
                file_text_parts.append(f"[Image {idx + 1}: {file_name}]")
            else:
                extracted = await fileE.extract_text(file_bytes_decoded, file_name)
                if extracted and len(extracted) > 500:
                    summary_prompt = (
                        "Summarise the key information, questions, and any given answers from the following content "
                        "in a concise way that preserves all important values, equations, and steps. "
                        "Do not explain or elaborate, just extract and compress:\n\n" + extracted
                    )
                    extracted = await ai.chat(site_id=user_id, system="You are a precise summariser.", user=summary_prompt)
                file_text_parts.append(f"=== File {idx + 1}: {file_name} ===\n{extracted}")
        except Exception as e:
            traceback.print_exc()
            await ws.send_json({"type": "error", "message": f"Could not read {file_name}: {e}"})

    if file_text_parts:
        file_context = "\n\n".join(file_text_parts)

    return file_context, image_attachments, False


def _build_history_lines(recent_history: list, max_chars: int = 30000) -> list:
    history_lines = []
    for m in recent_history:
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            if len(content) > max_chars:
                content = content[:max_chars]
            history_lines.append(f"User: {content}")
        elif role == "assistant":
            history_lines.append(f"Assistant: {content}")
    return history_lines


# ---------------------------------------------------------------------------
# Visual aids
# ---------------------------------------------------------------------------
async def _generate_visual(user_id: str, user_text: str, history_lines: list,
                           file_context: str, image_attachments: list) -> tuple:
    try:
        raw, d_in, d_out = await ai.get_diagram(
            site_id=user_id,
            user=user_text,
            conversation_context="\n".join(history_lines[-6:]),
            file_context=file_context,
            images=image_attachments or None,
        )
        if not raw:
            return None, d_in, d_out

        stripped = raw.strip()
        stripped = re.sub(r'^```\w*\n?', '', stripped)
        stripped = re.sub(r'\n?```$', '', stripped)
        upper = stripped.upper()

        if upper.startswith("NONE") or upper.startswith("INLINE"):
            return None, d_in, d_out
        if upper.startswith("MATH"):
            math_body = stripped[4:].strip().lstrip(":").strip()
            return ({"type": "math", "content": math_body} if math_body else None), d_in, d_out
        if upper.startswith("DIAGRAM"):
            match = re.search(r'<svg.*?</svg>', stripped, re.DOTALL | re.IGNORECASE)
            return ({"type": "diagram", "svg": match.group(0)} if match else None), d_in, d_out
        if upper.startswith("CODE"):
            body = stripped[4:].strip().lstrip(":").strip()
            lines = body.split("\n", 1)
            if len(lines) < 2:
                return None, d_in, d_out
            language = lines[0].strip().lower().replace("`", "")
            code_body = re.sub(r'^```\w*\n?', '', lines[1])
            code_body = re.sub(r'\n?```$', '', code_body).strip("\n")
            if not language or not code_body:
                return None, d_in, d_out
            return {"type": "code", "language": language, "code": code_body}, d_in, d_out
        if upper.startswith("HTML"):
            body = stripped[4:].strip().lstrip(":").strip()
            return ({"type": "html", "content": body} if body else None), d_in, d_out

        match = re.search(r'<svg.*?</svg>', stripped, re.DOTALL | re.IGNORECASE)
        return ({"type": "diagram", "svg": match.group(0)} if match else None), d_in, d_out

    except Exception as e:
        print(f"==> Visual aid generation failed: {e}")
        return None, 0, 0


async def _send_and_save_visual(ws, visual_aid, routing_decision: str, chat_id: str):
    if visual_aid is None:
        try:
            await ws.send_json({"type": "visual_aid_none"})
        except Exception:
            pass
        return

    vtype = visual_aid.get("type")
    if vtype == "diagram":
        await ws.send_json({"type": "diagram", "svg": visual_aid["svg"]})
    elif vtype == "code":
        await ws.send_json({"type": "code", "language": visual_aid["language"], "code": visual_aid["code"]})
    elif vtype == "math":
        await ws.send_json({"type": "math", "content": visual_aid["content"]})
    elif vtype == "html":
        await ws.send_json({"type": "html", "content": visual_aid["content"]})

    if vtype not in (None, "inline"):
        try:
            content_to_save = (
                visual_aid.get("svg")
                or visual_aid.get("code")
                or visual_aid.get("content", "")
            )
            await run_in_threadpool(
                rag.save_visual, chat_id, vtype, content_to_save, visual_aid.get("language")
            )
        except Exception as e:
            print(f"==> Failed to save visual: {e}")


async def _resolve_visual(visual_task) -> tuple:
    """Await an in-flight visual task → (visual_aid, d_in, d_out)."""
    if not visual_task:
        return None, 0, 0
    try:
        return await visual_task
    except Exception as e:
        print(f"==> Visual task error: {e}", flush=True)
        return None, 0, 0


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _build_integration_prompt(integrator, default_integration) -> str:
    """Pure prompt builder — caller supplies prefetched DB state (no I/O here)."""
    if not integrator:
        return "\n\nINTEGRATIONS RULE: User has not yet connected any services. You cannot perform calendar or external actions."

    if default_integration == "microsoft":
        return (
            "\n\nINTEGRATIONS: The user has connected Microsoft Calendar. "
            "You have the ability to read and write to their calendar.\n\n"
            "If the user asks to book, schedule, or create an event/meeting/appointment, "
            "respond naturally confirming what you're doing, then at the very end append:\n"
            "[CALENDAR_WRITE]{\"subject\": \"<title>\", \"start\": \"<ISO datetime>\", \"end\": \"<ISO datetime>\", \"body\": \"<optional notes>\", \"attendees\": []}\n\n"
            "If the user asks ANYTHING about their schedule, upcoming events, what they have planned, "
            "what's on their calendar, or questions like 'what do I have tomorrow', 'am I free at 3pm', "
            "'what's happening this week' respond naturally saying you're checking, then at the very end append:\n"
            "[CALENDAR_READ]{\"days_ahead\": <number based on the question, e.g. 1 for tomorrow, 7 for this week>}\n\n"
            "IMPORTANT: Only append a tag if a calendar action is clearly needed. "
            "If the user asks to cancel, delete, or remove an event:\n"
            "  Step 1 If you don't already have the event ID, respond naturally saying you're checking the calendar, then append:\n"
            "[CALENDAR_READ]{\"days_ahead\": 14}\n"
            "  Step 2 Once you have the event list and can identify the event the user wants to delete, append:\n"
            "[CALENDAR_DELETE]{\"event_id\": \"<event_id from the calendar data>\"}\n"
            "Never attempt a delete without a real event_id. If you cannot find the event, tell the user.\n\n"
            "For normal conversation, do not append any calendar tag. "
            "Do not use apostrophes in field values."
        )
    elif default_integration == "google":
        return (
            "\n\nINTEGRATIONS: The user has connected Google Calendar as their default provider. "
            "If the user asks to book, schedule, or create an event/meeting/appointment, respond naturally "
            "confirming what you're doing, then at the very end of your response append a calendar action "
            "in this exact format with no space between the tag and JSON:\n"
            "[GOOGLE_CALENDAR_WRITE]{\"subject\": \"<title>\", \"start\": \"<ISO datetime>\", \"end\": \"<ISO datetime>\", \"body\": \"<optional notes>\", \"attendees\": []}\n"
            "If the user asks to check, view, or list their calendar/events/schedule, respond naturally then append:\n"
            "[GOOGLE_CALENDAR_READ]{\"days_ahead\": 7}\n"
            "If the user asks to cancel, delete, or remove an event:\n"
            "  Step 1 If you don't already have the event ID, respond naturally saying you're checking the calendar, then append:\n"
            "[CALENDAR_READ]{\"days_ahead\": 14}\n"
            "  Step 2 Once you have the event list and can identify the event the user wants to delete, append:\n"
            "[CALENDAR_DELETE]{\"event_id\": \"<event_id from the calendar data>\"}\n"
            "Never attempt a delete without a real event_id. If you cannot find the event, tell the user.\n\n"
            "IMPORTANT: Only append the tag if a calendar action is clearly needed. "
            "For normal conversation, do not append any calendar tag. "
            "Do not use apostrophes in field values."
        )
    else:
        return "\n\nINTEGRATIONS RULE: User has connected services but has not set a default provider. Ask them to select one in settings."


def _build_websearch_prompt() -> str:
    return (
        "\n\nWEB SEARCH: You have access to real-time web search. "
        "If the user asks about current events, recent news, live data (weather, stocks, sports scores), "
        "or anything that requires up-to-date information beyond your training, "
        "respond naturally saying you're looking it up, then at the very end append:\n"
        "[WEB_SEARCH]{\"query\": \"<concise search query based on what the user asked>\"}\n\n"
        "IMPORTANT: Only use web search when the question genuinely requires current or real-time data. "
        "For general knowledge questions you already know the answer to, do not append a web search tag. "
        "Never mention that you're searching just say you're checking or looking it up."
    )


def _build_knowledge_graph_prompt(key_data: dict) -> str:
    try:
        if not key_data:
            return ""
        raw = key_data.get("knowledge_graph")
        if not raw:
            return ""
        graph = json.loads(raw)
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        node_map = {n["id"]: n for n in nodes}

        qa_pairs = []
        for edge in edges:
            source = node_map.get(edge.get("source"))
            target = node_map.get(edge.get("target"))
            if not source or not target:
                continue
            if source.get("type") == "question" and target.get("type") == "answer":
                q = source.get("data", {}).get("title", "").strip()
                a = target.get("data", {}).get("text", "").strip()
                if q and a:
                    qa_pairs.append((q, a))

        context_nodes = [
            n.get("data", {}).get("text", "").strip()
            for n in nodes if n.get("type") == "context" and n.get("data", {}).get("text", "").strip()
        ]
        rule_nodes = [
            n.get("data", {}).get("text", "").strip()
            for n in nodes if n.get("type") == "rule" and n.get("data", {}).get("text", "").strip()
        ]

        if not qa_pairs and not context_nodes and not rule_nodes:
            return ""

        lines = [
            "\n\nSTRUCTURED KNOWLEDGE BASE",
            "",
            "The business owner has explicitly defined the following knowledge for you to use. This is authoritative — treat it as ground truth. When a user asks something that matches a question below, use the paired answer directly. Do not paraphrase beyond what is needed for natural speech.",
        ]

        if qa_pairs:
            lines.append("")
            lines.append("QUESTION AND ANSWER PAIRS:")
            lines.append("When a user asks something similar to these questions, respond with the corresponding answer. You do not need an exact word-for-word match — use your judgement to match intent.")
            lines.append("")
            for q, a in qa_pairs:
                lines.append(f"  Q: {q}")
                lines.append(f"  A: {a}")
                lines.append("")

        if context_nodes:
            lines.append("BACKGROUND CONTEXT:")
            lines.append("This is always true about the business. Use it to inform your responses even when not directly asked.")
            lines.append("")
            for c in context_nodes:
                lines.append(f"  - {c}")
            lines.append("")

        if rule_nodes:
            lines.append("HARD RULES:")
            lines.append("These are non-negotiable instructions from the business owner. You must follow them at all times, no exceptions.")
            lines.append("")
            for r in rule_nodes:
                lines.append(f"  - {r}")
            lines.append("")

        lines.append("If a user asks something not covered by the above, fall back to the document knowledge base. If it is not there either, say you do not have that information.")
        return "\n".join(lines)

    except Exception as e:
        print(f"==> Knowledge graph prompt failed: {e}")
        return ""


def _build_availability_prompt(key_data: dict) -> str:
    try:
        if not key_data:
            return ""
        raw = key_data.get("availability")
        if not raw:
            return ""
        avail = json.loads(raw)

        day_names = {
            "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
            "thu": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday"
        }

        weekly = avail.get("weekly", {})
        overrides = avail.get("overrides", {})

        lines = [
            "\n\nAVAILABILITY & BOOKING RULES",
            "",
            "You have access to the business's availability schedule. Use this to answer questions about when appointments can be booked, whether a specific time is available, and what days the business operates.",
            "",
            "When a user asks to book an appointment, check whether their requested time falls within an available slot before confirming. If they ask a general question like 'when are you free' or 'can I come in Tuesday', answer directly using the schedule below.",
            "",
            "If a date has an override entry, that overrides the weekly default for that specific date.",
            "If a day shows no available slots, the business is closed or unavailable that day.",
            "",
            "WEEKLY SCHEDULE:",
        ]

        for day_key, day_name in day_names.items():
            ranges = weekly.get(day_key, [])
            if ranges:
                slots = ", ".join([f"{r['start']} to {r['end']}" for r in ranges])
                lines.append(f"  {day_name}: {slots}")
            else:
                lines.append(f"  {day_name}: Not available")

        if overrides:
            lines.append("")
            lines.append("DATE-SPECIFIC OVERRIDES (these take priority over the weekly schedule):")
            for date, override in overrides.items():
                ranges = override.get("ranges", [])
                if ranges:
                    slots = ", ".join([f"{r['start']} to {r['end']}" for r in ranges])
                    lines.append(f"  {date}: {slots}")
                else:
                    lines.append(f"  {date}: Closed (no availability this day)")

        lines.append("")
        lines.append("When booking, always confirm the exact time with the user before appending a CALENDAR_WRITE tag. Never book outside of available hours.")
        return "\n".join(lines)

    except Exception as e:
        print(f"==> Availability prompt failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# AI streaming
# ---------------------------------------------------------------------------
async def _stream_ai_response(
    ws, user_id: str, system_prompt: str, user_prompt: str,
    image_attachments: list, tts_queue,
) -> tuple:
    """Stream model deltas → frontend text_delta + sentence-chunked TTS queue.

    Time-to-first-audio optimisation: if the model's FIRST sentence is long,
    flush its first clause (at a comma) to TTS early so the avatar starts
    speaking while the rest of the sentence is still streaming. Applies only
    before any TTS has been queued — later sentences flush normally, so
    prosody is untouched for the bulk of the response.
    """
    full_text_parts = []
    sentence_buffer = ""
    sentences_for_tts = []
    chat_input_tokens = 0
    chat_output_tokens = 0
    tts_started = False

    async for delta in ai.stream(
        site_id=user_id,
        system=system_prompt,
        user=user_prompt,
        images=image_attachments or None,
    ):
        if delta.startswith("__USAGE__"):
            try:
                parts = delta[len("__USAGE__"):].split(",")
                chat_input_tokens += int(parts[0])
                chat_output_tokens += int(parts[1])
            except Exception:
                pass
            continue

        full_text_parts.append(delta)
        await ws.send_json({"type": "text_delta", "text": delta})

        sentence_buffer += delta
        while (m := SENTENCE_END_RE.search(sentence_buffer)):
            cut = m.end()
            sentence = sentence_buffer[:cut].strip()
            sentence_buffer = sentence_buffer[cut:]
            if sentence and len(sentence) > 2:
                sentences_for_tts.append(sentence)
                if tts_queue:
                    cleaned = _tts_clean(sentence)
                    if cleaned:
                        await tts_queue.put(cleaned)
                        tts_started = True

        # Early first-clause flush (first audio sooner on long openers)
        if (
            EARLY_FIRST_CLAUSE_FLUSH
            and tts_queue
            and not tts_started
            and len(sentence_buffer) >= EARLY_FLUSH_MIN_CHARS
        ):
            cut_at = sentence_buffer.rfind(", ")
            if cut_at >= EARLY_FLUSH_MIN_CLAUSE:
                clause = sentence_buffer[:cut_at + 1].strip()
                sentence_buffer = sentence_buffer[cut_at + 2:]
                sentences_for_tts.append(clause)
                cleaned = _tts_clean(clause)
                if cleaned:
                    await tts_queue.put(cleaned)
                    tts_started = True

    remaining = sentence_buffer.strip()
    remaining_clean = ACTION_TAG_PATTERN.sub('', remaining)
    remaining_clean = ROUTING_TAG_PATTERN.sub('', remaining_clean).strip()
    if remaining_clean and len(remaining_clean) > 2:
        sentences_for_tts.append(remaining)
        if tts_queue:
            cleaned = _tts_clean(remaining_clean)
            if cleaned:
                await tts_queue.put(cleaned)

    return full_text_parts, sentences_for_tts, chat_input_tokens, chat_output_tokens


async def _stream_second_pass(
    ws, user_id: str, user_text: str, context_text: str,
    system_prompt: str, voice_id: str, audio_on: bool,
) -> Optional[str]:
    """Second model pass after retrieving data (calendar / web). Returns bot text."""
    followup_system = (
        f"{system_prompt}\n\n"
        "You have just retrieved relevant data. Answer the user's question naturally and conversationally. "
        "Keep it brief, spoken aloud, under 80 words. No markdown, no tags."
    )
    followup_user = (
        f"The user asked: \"{user_text}\"\n\n"
        f"Here is the retrieved data:\n{context_text}\n\n"
        "Answer their question naturally based on this."
    )

    tts_queue, tts_task, tts_done_event = _start_tts(ws, voice_id, audio_on)

    try:
        full_text_parts, _, _, _ = await _stream_ai_response(
            ws, user_id, followup_system, followup_user, [], tts_queue
        )

        if tts_done_event:
            tts_done_event.set()

        raw_text = "".join(full_text_parts)
        bot_text = fix_markdown_formatting(_clean_bot_text(raw_text))

        await ws.send_json({"type": "text_done", "text": bot_text})
        await _await_tts(tts_task, "(second pass)")
        return bot_text

    except Exception as e:
        print(f"==> Second pass failed: {e}")
        traceback.print_exc()
        if tts_task:
            tts_task.cancel()
        return None


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------
async def _execute_calendar_action(
    ws, user_id: str, calendar_action: str, calendar_payload: dict,
    default_integration: str, user_text: str, system_prompt: str,
    voice_id: str, audio_on: bool,
) -> Optional[str]:
    """Execute the parsed calendar action. Returns the second-pass bot text
    (so the caller can persist it) or None."""
    is_google = (
        calendar_action in ("GOOGLE_CALENDAR_WRITE", "GOOGLE_CALENDAR_READ", "GOOGLE_CALENDAR_DELETE")
        or default_integration == "google"
    )

    if is_google:
        access_token = await google_get_token(user_id, rag)
        provider_name = "Google"
    else:
        access_token = await ms_get_token(user_id, rag)
        provider_name = "Microsoft"

    if not access_token:
        await ws.send_json({"type": "error", "message": f"{provider_name} not connected or token expired."})
        return None

    try:
        if calendar_action in ("CALENDAR_WRITE", "GOOGLE_CALENDAR_WRITE"):
            create_fn = google_create_event if is_google else ms_create_event
            result = await create_fn(
                access_token=access_token,
                subject=calendar_payload["subject"],
                start=datetime.fromisoformat(calendar_payload["start"].replace("Z", "+00:00")),
                end=datetime.fromisoformat(calendar_payload["end"].replace("Z", "+00:00")),
                body=calendar_payload.get("body", ""),
                attendee_emails=calendar_payload.get("attendees", []),
            )
            second_text = await _stream_second_pass(
                ws=ws, user_id=user_id, user_text=user_text,
                context_text=f"Event '{calendar_payload['subject']}' was successfully created.",
                system_prompt=system_prompt, voice_id=voice_id, audio_on=audio_on,
            )
            print(f"==> {provider_name} calendar event created: {result.get('id')}")
            await ws.send_json({"type": "calendar_done", "message": "Event booked!", "event": result})
            return second_text

        elif calendar_action in ("CALENDAR_READ", "GOOGLE_CALENDAR_READ"):
            list_fn = google_list_events if is_google else ms_list_events
            events = await list_fn(access_token=access_token, days_ahead=calendar_payload.get("days_ahead", 7))
            events_text = "\n".join([
                f"- {e.get('subject', 'Untitled')}: {e.get('start', {}).get('dateTime', '')} to {e.get('end', {}).get('dateTime', '')}"
                for e in events
            ]) if events else "No upcoming events found."
            print(f"==> {provider_name} calendar events fetched: {len(events)} events")
            return await _stream_second_pass(
                ws=ws, user_id=user_id, user_text=user_text,
                context_text=events_text,
                system_prompt=system_prompt, voice_id=voice_id, audio_on=audio_on,
            )

        elif calendar_action in ("CALENDAR_DELETE", "GOOGLE_CALENDAR_DELETE"):
            event_id = calendar_payload.get("event_id")
            if not event_id:
                await ws.send_json({"type": "error", "message": "No event ID provided to delete."})
                return None
            delete_fn = google_delete_event if is_google else ms_delete_event
            await delete_fn(access_token=access_token, event_id=event_id)
            await ws.send_json({"type": "calendar_done", "message": "Event deleted!"})
            return None

    except Exception as e:
        print(f"==> Calendar action failed: {e}")
        await ws.send_json({"type": "error", "message": f"Calendar action failed: {e}"})
        return None


# ---------------------------------------------------------------------------
# Summary / persistence
# ---------------------------------------------------------------------------
async def _update_summary(smgr, chat_id: str, history: list, user_text: str, bot_text: str):
    try:
        char_count = 0
        cutoff = len(history)
        for j in range(len(history) - 1, -1, -1):
            char_count += len(history[j].get("content", ""))
            if char_count > 800_000:
                cutoff = j + 1
                break
        recent = history[cutoff:].copy()
        recent.append({"role": "user", "content": user_text})
        recent.append({"role": "assistant", "content": bot_text})
        await smgr.on_new_message(chat_id, recent[-6:])
    except Exception as e:
        print(f"Summary update error: {e}")


def _save_turn(chat_id: str, bot_text: str, second_pass_text: Optional[str]):
    """Persist first-pass and (if present) second-pass assistant messages."""
    try:
        rag.add_message(chat_id=chat_id, role="assistant", content=bot_text)
        rag.update_last_message(chat_id=chat_id, last_message=bot_text)
    except Exception as e:
        print(f"==> Failed to save assistant message: {e}")

    if second_pass_text:
        try:
            rag.add_message(chat_id=chat_id, role="assistant", content=second_pass_text)
            rag.update_last_message(chat_id=chat_id, last_message=second_pass_text)
        except Exception as e:
            print(f"==> Failed to save second-pass message: {e}")


def _fire_and_forget(fn, *args):
    """Run a sync DB write off-loop without blocking the turn."""
    async def _run():
        try:
            await run_in_threadpool(fn, *args)
        except Exception as e:
            print(f"==> Background write failed ({getattr(fn, '__name__', fn)}): {e}")
    asyncio.create_task(_run())


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
@router.post("/chat_init")
async def chat_init(init_details: SessionInit, user=Depends(verify_token)):
    userID = init_details.userID
    chatID = init_details.chat_id

    avatar_key = voice_name = welcome_message = rive_url = prompt = ""
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
        "prompt": prompt,
    }


@router.post("/chat_diagram_init")
async def chat_diagram_init(init_details: DiagramInit, user=Depends(verify_token)):
    chatID = init_details.chat_id
    raw_visuals = rag.get_visuals(chatID)
    visuals = [
        {
            "visual_type": v.get("visual_type"),
            "content": v.get("content"),
            "language": v.get("language"),
            "created_at": str(v.get("created_at", "")),
        }
        for v in raw_visuals
    ]
    return {"visuals": visuals}


# ---------------------------------------------------------------------------
# MAIN APP WebSocket
# ---------------------------------------------------------------------------
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

            user_id   = _as_str(payload.get("user_id") or payload.get("site_id"))
            chat_id   = _as_str(payload.get("chat_id"))
            user_text = _as_str(payload.get("message"))
            voice_id  = _as_str(payload.get("voice_name"))
            prompt    = _as_str(payload.get("prompt"))
            audio_on  = payload.get("voice_on", True)
            pro_mode  = payload.get("pro_mode", False)
            raw_audio = payload.get("audio_bytes")

            files_payload = payload.get("files") or []
            if not files_payload:
                legacy_raw  = payload.get("file_bytes")
                legacy_name = payload.get("file_name")
                if legacy_raw and legacy_name:
                    files_payload = [{"file_bytes": legacy_raw, "file_name": legacy_name}]

            if raw_audio and "," in raw_audio:
                raw_audio = raw_audio.split(",", 1)[1]
            audio_bytes = base64.b64decode(raw_audio) if raw_audio else None

            chat_input_tokens = chat_output_tokens = 0
            diagram_input_tokens = diagram_output_tokens = 0

            if not user_id or not chat_id:
                await ws.send_json({"type": "error", "message": "Missing user_id/chat_id/message"})
                await ws.send_json({"type": "done"})
                continue

            if len(files_payload) > MAX_FILES:
                await ws.send_json({"type": "error", "message": f"Max {MAX_FILES} files per message.", "code": "TOO_MANY_FILES"})
                await ws.send_json({"type": "done"})
                continue

            # ── LATENCY: kick all per-turn DB reads in ONE threadpool hop
            #    NOW, so they overlap STT (300-800ms) instead of running
            #    serially after it. ──────────────────────────────────────
            smgr = get_summary_manager()
            prefetch = asyncio.create_task(run_in_threadpool(
                _prefetch_main_state, user_id, chat_id, not voice_id, smgr
            ))

            if audio_bytes:
                user_text = await _transcribe(ws, audio_bytes)
                if not user_text:
                    await _drain(prefetch)
                    await ws.send_json({"type": "done"})
                    continue

            if not user_text:
                await _drain(prefetch)
                await ws.send_json({"type": "error", "message": "Missing user_id/chat_id/message"})
                await ws.send_json({"type": "done"})
                continue

            state = await _drain(prefetch) or {}

            if not state.get("has_credits"):
                await ws.send_json({"type": "error", "message": "You have no credits remaining.", "code": "NO_CREDITS"})
                await ws.send_json({"type": "done"})
                continue

            # Save user message in the background — nothing downstream
            # reads it back this turn (history was already fetched).
            _fire_and_forget(rag.add_message, chat_id, "user", user_text)

            if not voice_id:
                avatar_data = state.get("avatar")
                if avatar_data:
                    voice_id = _as_str(avatar_data[1])
                if not voice_id:
                    voice_id = DEFAULT_VOICE_ID

            history = state.get("history") or []
            summary_context = state.get("summary")
            default_integration = state.get("default_integration")

            file_context, image_attachments, upload_too_large = await _process_files(ws, files_payload, user_id)
            if upload_too_large:
                await ws.send_json({"type": "done"})
                continue

            # ── system prompt ─────────────────────────────────────────
            system_prompt = f"{prompt}\n\n{summary_context}" if summary_context else prompt

            recent_history = history[-100:]
            history_lines = _build_history_lines(recent_history)
            conversation_history = "\n".join(history_lines)

            if conversation_history or summary_context:
                user_prompt = f"Conversation history:\n{conversation_history}\n\nLatest user message:\n{user_text}"
            else:
                user_prompt = user_text

            if file_context:
                system_prompt += (
                    f"\n\nThe user has attached {len(files_payload)} file(s). Here is the content:\n{file_context}"
                    "\n\nUse this as context. Do not read it verbatim. Explain conversationally."
                )
            else:
                system_prompt += "\n\nNo files attached."

            if audio_on:
                system_prompt += "\n\nLENGTH RULE: This response will be spoken aloud. Keep it under 120 words. Lead with the core answer, then the most important detail."
            else:
                system_prompt += "\n\nLENGTH RULE: Audio is off. You have room to be thorough. Use headers, lists, and examples freely."

            system_prompt += _build_integration_prompt(state.get("integrator"), default_integration)
            system_prompt += _build_websearch_prompt()

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            system_prompt += f"\n\nToday's date is {today} (UTC). The user is in Melbourne, Australia (AEST, UTC+10). When booking calendar events, use Melbourne local time."

            # ── first pass ────────────────────────────────────────────
            tts_queue, tts_task, tts_done_event = _start_tts(ws, voice_id, audio_on)
            search_task = None

            try:
                full_text_parts, sentences_for_tts, chat_input_tokens, chat_output_tokens = await _stream_ai_response(
                    ws, user_id, system_prompt, user_prompt, image_attachments, tts_queue
                )

                if tts_done_event:
                    tts_done_event.set()

                raw_text = "".join(full_text_parts)
                print(f"==> FULL RAW: {repr(raw_text)}", flush=True)

                calendar_action, calendar_payload = _detect_calendar_action(raw_text)
                web_match = _detect_web_search(raw_text)

                # LATENCY: start the web search NOW — it runs while the
                # avatar is still speaking the first-pass response, so the
                # 1-2s Tavily round-trip is hidden behind TTS playback.
                if web_match:
                    search_task = asyncio.create_task(run_in_threadpool(
                        get_web_search().web_search, web_match.group(1), 3
                    ))
                    await ws.send_json({"type": "web_search_pending"})

                bot_text = _clean_bot_text(raw_text)
                _, routing_decision = _extract_routing_tag(raw_text)
                bot_text = fix_markdown_formatting(bot_text)

                print(f"==> ROUTING: {routing_decision}", flush=True)

                if calendar_action:
                    await ws.send_json({"type": "calendar_action"})

                await ws.send_json({"type": "text_done", "text": bot_text})

                if not sentences_for_tts:
                    await ws.send_json({"type": "error", "message": "No response generated"})
                    await ws.send_json({"type": "done"})
                    if tts_task:
                        tts_task.cancel()
                    await _drain(search_task)
                    continue

            except Exception as e:
                await ws.send_json({"type": "error", "message": f"AI failed: {str(e)}"})
                await ws.send_json({"type": "done"})
                if tts_task:
                    tts_task.cancel()
                await _drain(search_task)
                continue

            # ── visual aid (concurrent with TTS) ──────────────────────
            visual_task = None
            if routing_decision in ("DIAGRAM", "CODE", "MATH", "HTML"):
                visual_task = asyncio.create_task(
                    _generate_visual(user_id, user_text, history_lines, file_context, image_attachments)
                )
                await ws.send_json({"type": "visual_aid_pending"})

            await _await_tts(tts_task)

            visual_aid, diagram_input_tokens, diagram_output_tokens = await _resolve_visual(visual_task)
            await _send_and_save_visual(ws, visual_aid, routing_decision, chat_id)

            # ── second pass (calendar / web) ──────────────────────────
            second_pass_text = None

            if calendar_action and calendar_payload:
                second_pass_text = await _execute_calendar_action(
                    ws=ws, user_id=user_id,
                    calendar_action=calendar_action, calendar_payload=calendar_payload,
                    default_integration=default_integration, user_text=user_text,
                    system_prompt=system_prompt, voice_id=voice_id, audio_on=audio_on,
                )

            if search_task:
                search_results = await _drain(search_task)
                second_pass_text = await _stream_second_pass(
                    ws=ws, user_id=user_id, user_text=user_text,
                    context_text=search_results or "(search failed)",
                    system_prompt=system_prompt, voice_id=voice_id, audio_on=audio_on,
                )

            # ── persist + summary ─────────────────────────────────────
            await run_in_threadpool(_save_turn, chat_id, bot_text, second_pass_text)
            await _update_summary(smgr, chat_id, history, user_text, second_pass_text or bot_text)

            # ── billing ───────────────────────────────────────────────
            try:
                cost = account_manager.processUsedCost(
                    input_tokens=chat_input_tokens + diagram_input_tokens,
                    output_tokens=chat_output_tokens + diagram_output_tokens,
                    SST_Length_seconds=len(audio_bytes) / STT_BYTES_PER_SECOND if audio_bytes else 0,
                    webSearch=bool(web_match),
                    voice_on=bool(audio_on),
                    diagram_on=bool(visual_aid),
                    pro_mode=bool(pro_mode),
                    image_count=len(image_attachments),
                    model=state.get("model") or "gemini",
                )
                credits_used = cost / AUD_PER_CREDIT
                remaining = await run_in_threadpool(rag.deductCredits, user_id, credits_used)
                print(f"==> Cost: ${cost:.4f} | Credits: {credits_used:.4f} | Remaining: {remaining}")
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


# ---------------------------------------------------------------------------
# EMBED WebSocket
# ---------------------------------------------------------------------------
@router.websocket("/embed_chat_ws")
async def embed_chat_ws(ws: WebSocket):
    t0 = time.time()
    print("HIT embed_chat_ws")
    await ws.accept()

    api_key = ws.query_params.get("api_key")
    if not api_key:
        await ws.close(code=4001, reason="Missing api_key")
        return

    initial_key_data = await run_in_threadpool(rag.getApiKey, api_key)
    if not initial_key_data or not initial_key_data.get("is_active"):
        await ws.close(code=4001, reason="Invalid or inactive API key")
        return

    owner_user_id = initial_key_data.get("owner_user_id")
    if not owner_user_id:
        await ws.close(code=4001, reason="API key has no owner")
        return

    print(f"==> [TIMING] key validated: {time.time()-t0:.2f}s", flush=True)

    # ── static config cached at connection open (incl. integrator state —
    #    no longer re-fetched from DB on every single turn) ──────────────
    cached_calendar_enabled = initial_key_data.get("calendar_enabled", True)
    cached_diagrams_enabled = bool(initial_key_data.get("embed_diagrams", False))
    cached_links_enabled    = bool(initial_key_data.get("link_enabled", False))

    cached_model, cached_default_integration, cached_has_docs, cached_integrator = await asyncio.gather(
        run_in_threadpool(rag.get_model, owner_user_id),
        run_in_threadpool(rag.getDefaultIntegration, owner_user_id),
        run_in_threadpool(rag.hasDocuments, api_key),
        run_in_threadpool(rag.checkIntegrations, owner_user_id),
    )
    if not cached_calendar_enabled:
        cached_default_integration = None

    cached_model = cached_model or "gemini"
    print(f"==> embed WS opened for api_key={api_key}, owner={owner_user_id}, model={cached_model}")

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

            # ── parse payload first so prefetch + STT can overlap ─────
            user_text  = _as_str(payload.get("message"))
            voice_id   = _as_str(payload.get("voice_name"))
            audio_on   = bool(payload.get("voice_on", True))
            raw_audio  = payload.get("audio_bytes")
            session_id = _as_str(payload.get("session_id")) or f"embed_{api_key}_{uuid.uuid4().hex[:12]}"

            asyncio.create_task(run_in_threadpool(
                rag.get_or_create_embed_session, session_id, api_key, owner_user_id
            ))

            if raw_audio and "," in raw_audio:
                raw_audio = raw_audio.split(",", 1)[1]
            audio_bytes = base64.b64decode(raw_audio) if raw_audio else None

            embed_input_tokens = embed_output_tokens = 0
            diagram_input_tokens = diagram_output_tokens = 0
            second_pass_text = None
            search_task = None

            # ── LATENCY: re-validate + all credit checks in ONE threadpool
            #    hop, overlapped with STT ─────────────────────────────────
            prefetch = asyncio.create_task(run_in_threadpool(
                _prefetch_embed_turn, api_key, owner_user_id
            ))

            if audio_bytes:
                user_text = await _transcribe(ws, audio_bytes)
                if not user_text:
                    await _drain(prefetch)
                    await ws.send_json({"type": "done"})
                    continue

            if not user_text:
                await _drain(prefetch)
                await ws.send_json({"type": "error", "message": "Missing message"})
                await ws.send_json({"type": "done"})
                continue

            turn = await _drain(prefetch) or {}
            key_data = turn.get("key_data")
            current_credits = turn.get("business_credits", 0.0)

            if not key_data or not key_data.get("is_active"):
                await ws.send_json({"type": "error", "message": "API key deactivated", "code": "INVALID_KEY"})
                await ws.close()
                return

            if key_data.get("conversations_used", 0) >= key_data.get("monthly_limit", 500):
                await ws.send_json({"type": "error", "message": "Monthly conversation limit reached", "code": "LIMIT_REACHED"})
                await ws.send_json({"type": "done"})
                continue

            if current_credits < MIN_CREDITS_PER_TURN:
                await ws.send_json({"type": "error", "message": "This business has run out of credits.", "code": "NO_CREDITS"})
                await ws.send_json({"type": "done"})
                continue

            if not turn.get("has_credits"):
                await ws.send_json({"type": "error", "message": "No credits remaining.", "code": "NO_CREDITS"})
                await ws.send_json({"type": "done"})
                continue

            # ── config ────────────────────────────────────────────────
            business_name        = key_data.get("business_name") or "this business"
            business_description = key_data.get("business_description") or ""
            assistant_name       = key_data.get("assistant_name") or "Assistant"
            version              = key_data.get("assistant_version", "professional")

            if not voice_id:
                avatar = await run_in_threadpool(
                    rag.getAvatarByName, key_data.get("avatar_name") or "Mia Sterling"
                ) or {}
                voice_id = avatar.get("voice") or DEFAULT_VOICE_ID

            # ── history (load + save user msg in parallel) ────────────
            try:
                history, _ = await asyncio.gather(
                    run_in_threadpool(rag.get_recent_messages, owner_user_id, session_id, 10),
                    run_in_threadpool(rag.add_message, session_id, "user", user_text),
                )
            except Exception as e:
                print(f"==> History/save failed: {e}")
                history = []

            print(f"==> [TIMING] pre-RAG done: {time.time()-t0:.2f}s", flush=True)

            # ── RAG (embed + vector search, both off-loop) ────────────
            rag_context = ""
            if cached_has_docs:
                try:
                    embedding = await rag.embedText(user_text)
                    chunks = await run_in_threadpool(
                        rag.searchDocumentChunks, api_key, embedding, 5
                    )
                    print(f"==> RAG chunks: {len(chunks)}", flush=True)
                    for c in chunks:
                        print(f"==>   sim={c.get('similarity',0):.3f} | {c.get('content','')[:80]}", flush=True)
                    if chunks and chunks[0].get("similarity", 0) >= 0.2:
                        rag_context = "\n".join(f"- {c['content']}" for c in chunks)
                except Exception as e:
                    print(f"==> RAG failed: {e}")
            else:
                print("==> RAG skipped — no docs", flush=True)

            print(f"==> [TIMING] RAG done: {time.time()-t0:.2f}s", flush=True)

            # ── system prompt ─────────────────────────────────────────
            tone_map = {
                "professional": "You are polite, professional, and clear. Sound like a well-trained support agent.",
                "friendly": "You are warm, casual, and approachable. Sound like a helpful friend who works at the business. Use informal language, contractions, and a light conversational tone.",
                "concise": "You are extremely brief. One to two sentences max. No filler. Just the answer.",
            }
            tone = tone_map.get(version, tone_map["professional"])
            kb_section = rag_context if rag_context else "No specific documents loaded."

            system_prompt = (
                f"You are {assistant_name}, a support assistant for {business_name}. "
                f"{business_description}\n\n"

                "YOUR PERSONALITY\n\n"
                f"{tone} You sound like a real person who works at the business, not a corporate chatbot. "
                "Use contractions. Be concise but friendly.\n\n"

                "WHAT YOU CAN HELP WITH\n\n"
                f"1. Answer questions about {business_name} using the knowledge base below. This is your primary source of truth.\n\n"
                "2. For general factual questions that are not business-specific, you may use your general knowledge to help — "
                f"but never invent facts about {business_name} itself.\n\n"
                f"3. If someone asks about the business and the answer is not in your knowledge base, be honest and suggest "
                f"they contact the business directly. Never make up business-specific details like hours, prices, policies, or contact info.\n\n"

                "CONVERSATION RULES\n\n"
                "4. Keep responses under 80 words. This is spoken aloud, not read on screen. Short sentences work best.\n\n"
                "5. No markdown, no bullet points, no asterisks, no headers. Plain conversational sentences only.\n\n"
                "6. Remember what was said earlier in this conversation and refer back to it naturally when relevant. "
                "If someone asks what they asked before, recap it briefly.\n\n"
                "7. Never say 'based on my training', 'as an AI', or 'I believe'. Just answer naturally.\n\n"
                f"8. If someone asks who you are, say: I'm {assistant_name}, here to help with {business_name}.\n\n"
                "9. Never make up phone numbers, addresses, hours, staff names, or prices unless they are in the knowledge base.\n\n"
                "10. If someone asks something completely off-topic and unrelated to the business or general helpful advice, "
                "politely redirect them.\n\n"

                f"KNOWLEDGE BASE — {business_name.upper()}\n"
                f"The following is verified information about {business_name}. Trust this above anything else.\n\n"
                f"{kb_section}"
            )

            system_prompt += _build_knowledge_graph_prompt(key_data)

            if cached_calendar_enabled:
                system_prompt += _build_availability_prompt(key_data)
                system_prompt += _build_integration_prompt(cached_integrator, cached_default_integration)
            else:
                system_prompt += "\n\nCALENDAR: This business has not enabled calendar integrations. Do not offer booking or calendar actions."

            system_prompt += _build_websearch_prompt()

            if cached_diagrams_enabled:
                system_prompt += (
                    "\n\nVISUAL AIDS: You can generate visual aids. End your response with one of these tags:\n"
                    "[NONE] - no visual needed\n"
                    "[DIAGRAM] - flowchart or architecture diagram\n"
                    "[CODE] - code snippet\n"
                    "[MATH] - equation or formula\n"
                    "[HTML] - interactive visual\n"
                    "Only use a visual tag if it genuinely helps. Default to [NONE]."
                )

            if cached_links_enabled:
                system_prompt += (
                    "\n\nLINK SHARING: You can share a relevant link with the user when it would genuinely help. "
                    "For example if they ask about booking, directions, a product page, or anything the business has a URL for. "
                    "If you have a relevant link from the knowledge base, end your response with:\n"
                    "[LINK]{\"url\": \"<full url>\", \"label\": \"<short label like Book Now or Get Directions>\"}\n"
                    "Only include a link if it is directly relevant. Never make up URLs."
                )

            if audio_on:
                system_prompt += "\n\nLENGTH RULE: This response will be spoken aloud. Keep it under 80 words. Lead with the core answer."

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            system_prompt += f"\n\nToday's date is {today} (UTC). The user is in Melbourne, Australia (AEST, UTC+10). When booking calendar events, use Melbourne local time."

            # ── user prompt ───────────────────────────────────────────
            history_lines = _build_history_lines(history[-6:])
            if history_lines:
                user_prompt = "Conversation history:\n" + "\n".join(history_lines) + f"\n\nLatest user message:\n{user_text}"
            else:
                user_prompt = user_text

            # ── first pass ────────────────────────────────────────────
            tts_queue, tts_task, tts_done_event = _start_tts(ws, voice_id, audio_on)

            try:
                print(f"==> [TIMING] AI start: {time.time()-t0:.2f}s", flush=True)

                full_text_parts, sentences_for_tts, embed_input_tokens, embed_output_tokens = await _stream_ai_response(
                    ws, owner_user_id, system_prompt, user_prompt, [], tts_queue
                )

                if tts_done_event:
                    tts_done_event.set()

                raw_text = "".join(full_text_parts).strip()
                print(f"==> embed FULL RAW: {repr(raw_text)}", flush=True)

                calendar_action, calendar_payload_data = _detect_calendar_action(raw_text)
                web_match = _detect_web_search(raw_text)
                link_match = _detect_link(raw_text) if cached_links_enabled else None

                # LATENCY: start search while TTS is still speaking
                if web_match:
                    search_task = asyncio.create_task(run_in_threadpool(
                        get_web_search().web_search, web_match.group(1), 3
                    ))
                    await ws.send_json({"type": "web_search_pending"})

                _, routing_decision = _extract_routing_tag(raw_text)
                if not cached_diagrams_enabled:
                    routing_decision = "NONE"

                bot_text = _clean_bot_text(raw_text)
                if link_match:
                    bot_text = LINK_TAG_RE.sub('', bot_text).strip()
                bot_text = fix_markdown_formatting(bot_text)

                if calendar_action:
                    await ws.send_json({"type": "calendar_action"})
                if link_match:
                    try:
                        link_data = json.loads(link_match.group(1))
                        await ws.send_json({
                            "type": "link",
                            "url": link_data.get("url", ""),
                            "label": link_data.get("label", "Open Link"),
                        })
                    except Exception as e:
                        print(f"==> Link send failed: {e}")

                await ws.send_json({"type": "text_done", "text": bot_text})
                print(f"==> [TIMING] AI done: {time.time()-t0:.2f}s", flush=True)

                if not sentences_for_tts:
                    await ws.send_json({"type": "error", "message": "No response generated"})
                    await ws.send_json({"type": "done"})
                    if tts_task:
                        tts_task.cancel()
                    await _drain(search_task)
                    continue

            except Exception as e:
                await ws.send_json({"type": "error", "message": "The assistant is unavailable right now. Please try again shortly."})
                print(f"==> embed AI failed: {e}")
                traceback.print_exc()
                if tts_task:
                    tts_task.cancel()
                await _drain(search_task)
                await ws.send_json({"type": "done"})
                continue

            # ── visual aid (concurrent with TTS) ──────────────────────
            visual_task = None
            if cached_diagrams_enabled and routing_decision in ("DIAGRAM", "CODE", "MATH", "HTML"):
                visual_task = asyncio.create_task(
                    _generate_visual(owner_user_id, user_text, history_lines, "", [])
                )
                await ws.send_json({"type": "visual_aid_pending"})

            await _await_tts(tts_task, "(embed)")
            print(f"==> [TIMING] TTS done: {time.time()-t0:.2f}s", flush=True)

            visual_aid, diagram_input_tokens, diagram_output_tokens = await _resolve_visual(visual_task)
            if cached_diagrams_enabled:
                await _send_and_save_visual(ws, visual_aid, routing_decision, session_id)
            else:
                try:
                    await ws.send_json({"type": "visual_aid_none"})
                except Exception:
                    pass

            # ── second pass (calendar / web) ──────────────────────────
            if cached_calendar_enabled and calendar_action and calendar_payload_data:
                second_pass_text = await _execute_calendar_action(
                    ws=ws, user_id=owner_user_id,
                    calendar_action=calendar_action, calendar_payload=calendar_payload_data,
                    default_integration=cached_default_integration, user_text=user_text,
                    system_prompt=system_prompt, voice_id=voice_id, audio_on=audio_on,
                )

            if search_task:
                search_results = await _drain(search_task)
                second_pass_text = await _stream_second_pass(
                    ws=ws, user_id=owner_user_id, user_text=user_text,
                    context_text=search_results or "(search failed)",
                    system_prompt=system_prompt, voice_id=voice_id, audio_on=audio_on,
                )

            # ── persist (off-loop) ────────────────────────────────────
            await run_in_threadpool(_save_turn, session_id, bot_text, second_pass_text)

            # ── billing ───────────────────────────────────────────────
            _fire_and_forget(rag.incrementConversationCount, api_key)

            try:
                cost_aud = account_manager.processUsedCost(
                    input_tokens=embed_input_tokens + diagram_input_tokens,
                    output_tokens=embed_output_tokens + diagram_output_tokens,
                    outputText=bot_text,
                    SST_Length_seconds=len(audio_bytes) / STT_BYTES_PER_SECOND if audio_bytes else 0,
                    webSearch=bool(web_match),
                    voice_on=bool(audio_on),
                    diagram_on=bool(visual_aid),
                    model=cached_model,
                )
                _fire_and_forget(rag.addApiKeyCost, api_key, cost_aud)

                try:
                    new_balance = await run_in_threadpool(
                        rag.deductBusinessCredits, owner_user_id, cost_aud
                    )
                    print(f"==> Embed cost: ${cost_aud:.4f} | balance: ${new_balance:.4f}", flush=True)
                    if new_balance < 0.50:
                        await ws.send_json({"type": "credits_low", "balance": new_balance, "message": "Credits running low"})
                except Exception as e:
                    print(f"==> Failed to deduct business credits: {e}")

            except Exception as e:
                print(f"==> Embed cost tracking failed: {e}", flush=True)

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