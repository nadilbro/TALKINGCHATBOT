import re
import asyncio
from html import unescape
import traceback
from typing import Any
import base64
import time
import uuid
import json

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
from datetime import datetime, timezone

router = APIRouter(prefix="/system", tags=["chat"])

rag = VectorRAGService()
ai = AIProvider(rag)
account_manager = AccountManager(rag)
fileE = FileExtractor(ai)

MINUTES_PER_CREDIT = 7

ACTION_TAG_PATTERN = re.compile(
    r'\[?(CALENDAR_WRITE|CALENDAR_READ|CALENDAR_DELETE|'
    r'GOOGLE_CALENDAR_WRITE|GOOGLE_CALENDAR_READ|GOOGLE_CALENDAR_DELETE|'
    r'WEB_SEARCH)\]?\s*\{[^}]*\}',
    re.IGNORECASE
)

ROUTING_TAG_PATTERN = re.compile(
    r'\[?(NONE|INLINE|DIAGRAM|CODE|MATH|HTML)\]?\s*$',
    re.IGNORECASE
)

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


def _as_str(x: Any) -> str:
    return (str(x) if x is not None else "").strip()


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


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


def _extract_routing_tag(text: str) -> tuple:
    match = re.search(r'\[(NONE|INLINE|DIAGRAM|CODE|MATH|HTML)\]\s*$', text, re.IGNORECASE)
    if match:
        tag = match.group(1).upper()
        cleaned = text[:match.start()].rstrip()
        return cleaned, tag
    lines = text.strip().rsplit('\n', 1)
    if len(lines) == 2:
        last_line = lines[1].strip().upper()
        if last_line in ('NONE', 'INLINE', 'DIAGRAM', 'CODE', 'MATH', 'HTML',
                         '[NONE]', '[INLINE]', '[DIAGRAM]', '[CODE]', '[MATH]', '[HTML]'):
            tag = last_line.strip('[]')
            return lines[0].rstrip(), tag
    return text, "NONE"


def _detect_calendar_action(raw_text: str) -> tuple:
    cal_match = re.search(
        r'\[(CALENDAR_WRITE|CALENDAR_READ|CALENDAR_DELETE|'
        r'GOOGLE_CALENDAR_WRITE|GOOGLE_CALENDAR_READ|GOOGLE_CALENDAR_DELETE)\](\{[^}]*\})',
        raw_text
    )
    if not cal_match:
        return None, None
    action = cal_match.group(1)
    try:
        payload = json.loads(cal_match.group(2))
        return action, payload
    except Exception as e:
        print(f"==> Failed to parse calendar JSON: {e} | raw: {repr(cal_match.group(2))}")
        return None, None


def _detect_web_search(raw_text: str):
    return re.search(r'\[WEB_SEARCH\]\{"query":\s*"([^"]+)"\}', raw_text)


def _clean_bot_text(raw_text: str) -> str:
    """Strips ALL action tags, routing tags, and artifacts before sending to frontend."""
    text = ACTION_TAG_PATTERN.sub('', raw_text)
    text, _ = _extract_routing_tag(text)
    text = re.sub(r'```[a-z]*\n?.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def _tts_clean(sentence: str) -> str:
    """Cleans a sentence before sending to ElevenLabs. Strips all tags."""
    ROUTING_TAGS = {'none', 'inline', 'diagram', 'code', 'math', 'html'}
    cleaned = strip_markdown(sentence)
    cleaned = ACTION_TAG_PATTERN.sub('', cleaned)
    cleaned = ROUTING_TAG_PATTERN.sub('', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned or len(cleaned) < 3:
        return ''
    if cleaned.lower() in ROUTING_TAGS:
        return ''
    return cleaned


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

            result = await tts_instance.synthesize_sentence(cleaned, voice_id)
            if isinstance(result, Exception):
                print(f"==> TTS sentence failed: {result}", flush=True)
                continue

            audio_bytes_chunk, visemes, duration = result
            if not audio_bytes_chunk:
                continue

            chunk_visemes = [
                {"t_ms": v["t_ms"] + cumulative_offset_ms, "viseme_id": v["viseme_id"]}
                for v in visemes
            ]

            if first_chunk:
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
                if chunk_visemes:
                    await ws.send_json({
                        "type": "viseme_update",
                        "visemes": [
                            {"t_ms": _as_int(v.get("t_ms"), 0), "viseme_id": _as_int(v.get("viseme_id"), 0)}
                            for v in chunk_visemes
                        ],
                    })

            CHUNK = 32_000
            for i in range(0, len(audio_bytes_chunk), CHUNK):
                await ws.send_bytes(audio_bytes_chunk[i:i + CHUNK])

            cumulative_offset_ms += int(duration * 1000)

        except Exception as e:
            print(f"==> TTS pipeline error: {e}", flush=True)
            traceback.print_exc()

    if not first_chunk:
        try:
            await ws.send_json({"type": "audio_end"})
        except Exception:
            pass


async def _process_files(ws, files_payload: list, user_id: str) -> tuple:
    MAX_TOTAL_UPLOAD_BYTES = 40 * 1024 * 1024
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
        except Exception as e:
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
                mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
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


async def _generate_visual(user_id: str, user_text: str, history_lines: list, file_context: str, image_attachments: list) -> tuple:
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

    if visual_aid["type"] == "diagram":
        await ws.send_json({"type": "diagram", "svg": visual_aid["svg"]})
    elif visual_aid["type"] == "code":
        await ws.send_json({"type": "code", "language": visual_aid["language"], "code": visual_aid["code"]})
    elif visual_aid["type"] == "math":
        await ws.send_json({"type": "math", "content": visual_aid["content"]})
    elif visual_aid["type"] == "html":
        await ws.send_json({"type": "html", "content": visual_aid["content"]})

    if visual_aid.get("type") not in (None, "inline"):
        try:
            content_to_save = (
                visual_aid.get("svg")
                or visual_aid.get("code")
                or visual_aid.get("content", "")
                or visual_aid.get("math")
                or visual_aid.get("html")
            )
            rag.save_visual(
                session_id=chat_id,
                visual_type=visual_aid["type"],
                content=content_to_save,
                language=visual_aid.get("language"),
            )
        except Exception as e:
            print(f"==> Failed to save visual: {e}")


def _build_integration_prompt(user_id: str) -> str:
    integrators = rag.checkIntegrations(user_id)
    if not integrators:
        return "\n\nINTEGRATIONS RULE: User has not yet connected any services. You cannot perform calendar or external actions."

    default_integration = rag.getDefaultIntegration(user_id)
    print(f"==> DEFAULT INTEGRATION: {default_integration} for {user_id}")
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


async def _execute_calendar_action(
    ws, user_id: str, calendar_action: str, calendar_payload: dict,
    default_integration: str, user_text: str, system_prompt: str,
    voice_id: str, audio_on: bool
):
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
        return

    try:
        if calendar_action in ("CALENDAR_WRITE", "GOOGLE_CALENDAR_WRITE"):
            if is_google:
                result = await google_create_event(
                    access_token=access_token,
                    subject=calendar_payload["subject"],
                    start=datetime.fromisoformat(calendar_payload["start"].replace("Z", "+00:00")),
                    end=datetime.fromisoformat(calendar_payload["end"].replace("Z", "+00:00")),
                    body=calendar_payload.get("body", ""),
                    attendee_emails=calendar_payload.get("attendees", []),
                )
            else:
                result = await ms_create_event(
                    access_token=access_token,
                    subject=calendar_payload["subject"],
                    start=datetime.fromisoformat(calendar_payload["start"].replace("Z", "+00:00")),
                    end=datetime.fromisoformat(calendar_payload["end"].replace("Z", "+00:00")),
                    body=calendar_payload.get("body", ""),
                    attendee_emails=calendar_payload.get("attendees", []),
                )
            await _stream_second_pass(
                ws=ws, user_id=user_id, user_text=user_text,
                context_text=f"Event '{calendar_payload['subject']}' was successfully created.",
                system_prompt=system_prompt, voice_id=voice_id, audio_on=audio_on,
            )
            print(f"==> {provider_name} calendar event created: {result.get('id')}")
            await ws.send_json({"type": "calendar_done", "message": "Event booked!", "event": result})

        elif calendar_action in ("CALENDAR_READ", "GOOGLE_CALENDAR_READ"):
            if is_google:
                events = await google_list_events(access_token=access_token, days_ahead=calendar_payload.get("days_ahead", 7))
            else:
                events = await ms_list_events(access_token=access_token, days_ahead=calendar_payload.get("days_ahead", 7))

            events_text = "\n".join([
                f"- {e.get('subject', 'Untitled')}: {e.get('start', {}).get('dateTime', '')} to {e.get('end', {}).get('dateTime', '')}"
                for e in events
            ]) if events else "No upcoming events found."

            print(f"==> {provider_name} calendar events fetched: {len(events)} events")
            await _stream_second_pass(
                ws=ws, user_id=user_id, user_text=user_text,
                context_text=events_text,
                system_prompt=system_prompt, voice_id=voice_id, audio_on=audio_on,
            )

        elif calendar_action in ("CALENDAR_DELETE", "GOOGLE_CALENDAR_DELETE"):
            event_id = calendar_payload.get("event_id")
            if not event_id:
                await ws.send_json({"type": "error", "message": "No event ID provided to delete."})
                return
            if is_google:
                await google_delete_event(access_token=access_token, event_id=event_id)
            else:
                await ms_delete_event(access_token=access_token, event_id=event_id)
            await ws.send_json({"type": "calendar_done", "message": "Event deleted!"})

    except Exception as e:
        print(f"==> Calendar action failed: {e}")
        await ws.send_json({"type": "error", "message": f"Calendar action failed: {e}"})


async def _stream_second_pass(
    ws, user_id: str, user_text: str, context_text: str,
    system_prompt: str, voice_id: str, audio_on: bool,
    context_label: str = "data"
) -> str | None:
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

    tts_queue = None
    tts_task = None
    tts_done_event = None

    try:
        if audio_on:
            tts_queue = asyncio.Queue()
            tts_done_event = asyncio.Event()
            tts_task = asyncio.create_task(_tts_pipeline(ws, tts_queue, voice_id, tts_done_event))

        full_text_parts, _, _, _ = await _stream_ai_response(
            ws, user_id, followup_system, followup_user,
            [], audio_on, voice_id, tts_queue, False
        )

        if tts_done_event:
            tts_done_event.set()

        raw_text = "".join(full_text_parts)
        bot_text = _clean_bot_text(raw_text)
        bot_text = fix_markdown_formatting(bot_text)

        await ws.send_json({"type": "text_done", "text": bot_text})

        if tts_task:
            try:
                await asyncio.wait_for(tts_task, timeout=30)
            except asyncio.TimeoutError:
                tts_task.cancel()
            except Exception as e:
                print(f"==> Second pass TTS error: {e}")

        return bot_text

    except Exception as e:
        print(f"==> Second pass failed: {e}")
        traceback.print_exc()
        if tts_task:
            tts_task.cancel()
        return None

async def _update_summary(smgr, chat_id: str, history: list, user_text: str, bot_text: str):
    try:
        char_count = 0
        cutoff = len(history)
        for j in range(len(history) - 1, -1, -1):
            char_count += len(history[j].get("content", ""))
            if char_count > 800_000:
                cutoff = j + 1
                break
        trimmed = history[cutoff:]
        recent = trimmed.copy()
        recent.append({"role": "user", "content": user_text})
        recent.append({"role": "assistant", "content": bot_text})
        await smgr.on_new_message(chat_id, recent[-6:])
    except Exception as e:
        print(f"Summary update error: {e}")


async def _stream_ai_response(
    ws, user_id: str, system_prompt: str, user_prompt: str,
    image_attachments: list, audio_on: bool, voice_id: str,
    tts_queue, routing_tag_found: bool
) -> tuple:
    full_text_parts = []
    sentence_buffer = ""
    sentences_for_tts = []
    chat_input_tokens = 0
    chat_output_tokens = 0

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
        while re.search(r'[.?!]\s', sentence_buffer):
            match = re.search(r'[.?!]\s', sentence_buffer)
            cut = match.end()
            sentence = sentence_buffer[:cut].strip()
            sentence_buffer = sentence_buffer[cut:]
            if sentence and len(sentence) > 2:
                sentences_for_tts.append(sentence)
                if tts_queue and not routing_tag_found:
                    cleaned = _tts_clean(sentence)
                    if cleaned:
                        await tts_queue.put(cleaned)

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
# -----------------------------------------------------------------------
# KNOWLEDGE GRAPH PROMPT BUILDER
# -----------------------------------------------------------------------
def _build_knowledge_graph_prompt(api_key: str) -> str:
    """Converts the knowledge graph JSON into a structured prompt section."""
    try:
        key_data = rag.getApiKey(api_key)
        if not key_data:
            return ""
        raw = key_data.get("knowledge_graph")
        if not raw:
            return ""
        graph = json.loads(raw)
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        # Build lookup
        node_map = {n["id"]: n for n in nodes}

        # Find connected Q->A pairs
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

        # Context nodes (always active)
        context_nodes = [
            n.get("data", {}).get("text", "").strip()
            for n in nodes if n.get("type") == "context"
        ]

        # Rule nodes
        rule_nodes = [
            n.get("data", {}).get("text", "").strip()
            for n in nodes if n.get("type") == "rule"
        ]

        sections = []

        if qa_pairs:
            qa_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])
            sections.append(f"STRUCTURED Q&A:\n{qa_text}")

        if context_nodes:
            ctx_text = "\n".join([f"- {c}" for c in context_nodes if c])
            sections.append(f"BACKGROUND CONTEXT:\n{ctx_text}")

        if rule_nodes:
            rule_text = "\n".join([f"- {r}" for r in rule_nodes if r])
            sections.append(f"HARD RULES:\n{rule_text}")

        if not sections:
            return ""

        return "\n\n" + "\n\n".join(sections)

    except Exception as e:
        print(f"==> Knowledge graph prompt failed: {e}")
        return ""


# -----------------------------------------------------------------------
# AVAILABILITY PROMPT BUILDER
# -----------------------------------------------------------------------
def _build_availability_prompt(api_key: str) -> str:
    """Converts the availability JSON into a readable prompt section."""
    try:
        key_data = rag.getApiKey(api_key)
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

        lines = ["AVAILABILITY:"]

        for day_key, day_name in day_names.items():
            ranges = weekly.get(day_key, [])
            if ranges:
                slots = ", ".join([f"{r['start']} to {r['end']}" for r in ranges])
                lines.append(f"  {day_name}: {slots}")
            else:
                lines.append(f"  {day_name}: Closed")

        if overrides:
            lines.append("DATE OVERRIDES:")
            for date, override in overrides.items():
                ranges = override.get("ranges", [])
                if ranges:
                    slots = ", ".join([f"{r['start']} to {r['end']}" for r in ranges])
                    lines.append(f"  {date}: {slots}")
                else:
                    lines.append(f"  {date}: Closed")

        return "\n" + "\n".join(lines)

    except Exception as e:
        print(f"==> Availability prompt failed: {e}")
        return ""

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

            user_id    = _as_str(payload.get("user_id") or payload.get("site_id"))
            chat_id    = _as_str(payload.get("chat_id"))
            user_text  = _as_str(payload.get("message"))
            voice_id   = _as_str(payload.get("voice_name"))
            prompt     = _as_str(payload.get("prompt"))
            audio_on   = payload.get("voice_on", True)
            pro_mode   = payload.get("pro_mode", False)
            raw_audio  = payload.get("audio_bytes")

            files_payload = payload.get("files") or []
            if not files_payload:
                legacy_raw  = payload.get("file_bytes")
                legacy_name = payload.get("file_name")
                if legacy_raw and legacy_name:
                    files_payload = [{"file_bytes": legacy_raw, "file_name": legacy_name}]

            MAX_FILES = 5
            if raw_audio and "," in raw_audio:
                raw_audio = raw_audio.split(",", 1)[1]
            audio_bytes = base64.b64decode(raw_audio) if raw_audio else None

            chat_input_tokens = chat_output_tokens = 0
            diagram_input_tokens = diagram_output_tokens = 0

            if not rag.hasEnoughCredits(user_id):
                await ws.send_json({"type": "error", "message": "You have no credits remaining.", "code": "NO_CREDITS"})
                await ws.send_json({"type": "done"})
                continue

            if len(files_payload) > MAX_FILES:
                await ws.send_json({"type": "error", "message": f"Max {MAX_FILES} files per message.", "code": "TOO_MANY_FILES"})
                await ws.send_json({"type": "done"})
                continue

            if audio_bytes:
                try:
                    user_text = get_stt().get_transcript(audio_bytes)
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

            try:
                rag.add_message(chat_id=chat_id, role="user", content=user_text)
            except Exception as e:
                print(f"==> Failed to save user message: {e}")

            if not voice_id:
                try:
                    avatar_data = rag.get_avatar(user_id, chat_id)
                    if avatar_data:
                        voice_id = _as_str(avatar_data[1])
                except Exception:
                    pass
                if not voice_id:
                    voice_id = "UgBBYS2sOqTuMpoF3BR0"

            try:
                history = rag.get_recent_messages(user_id=user_id, chat_id=chat_id, limit=20)
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"Failed to load history: {str(e)}"})
                continue

            file_context, image_attachments, upload_too_large = await _process_files(ws, files_payload, user_id)
            if upload_too_large:
                await ws.send_json({"type": "done"})
                continue

            smgr = get_summary_manager()
            summary_context = smgr.build_context(chat_id)
            system_prompt = f"{prompt}\n\n{summary_context}" if summary_context else prompt

            recent_history = history[-100:] if len(history) < 100 else history
            history_lines = _build_history_lines(recent_history)
            conversation_history = "\n".join(history_lines)

            if conversation_history or summary_context:
                user_prompt = f"Conversation history:\n{conversation_history}\n\nLatest user message:\n{user_text}"
            else:
                user_prompt = user_text

            if file_context:
                system_prompt += f"\n\nThe user has attached {len(files_payload)} file(s). Here is the content:\n{file_context}\n\nUse this as context. Do not read it verbatim. Explain conversationally."
            else:
                system_prompt += "\n\nNo files attached."

            if audio_on:
                system_prompt += "\n\nLENGTH RULE: This response will be spoken aloud. Keep it under 120 words. Lead with the core answer, then the most important detail."
            else:
                system_prompt += "\n\nLENGTH RULE: Audio is off. You have room to be thorough. Use headers, lists, and examples freely."

            system_prompt += _build_integration_prompt(user_id)
            system_prompt += _build_websearch_prompt()

            default_integration = rag.getDefaultIntegration(user_id)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            system_prompt += f"\n\nToday's date is {today} (UTC). The user is in Melbourne, Australia (AEST, UTC+10). When booking calendar events, use Melbourne local time."

            tts_queue = tts_task = tts_done_event = None

            try:
                if audio_on:
                    tts_queue = asyncio.Queue()
                    tts_done_event = asyncio.Event()
                    tts_task = asyncio.create_task(_tts_pipeline(ws, tts_queue, voice_id, tts_done_event))

                full_text_parts, sentences_for_tts, chat_input_tokens, chat_output_tokens = await _stream_ai_response(
                    ws, user_id, system_prompt, user_prompt,
                    image_attachments, audio_on, voice_id, tts_queue, False
                )

                if tts_done_event:
                    tts_done_event.set()

                raw_text = "".join(full_text_parts)
                print(f"==> FULL RAW: {repr(raw_text)}", flush=True)

                calendar_action, calendar_payload = _detect_calendar_action(raw_text)
                web_match = _detect_web_search(raw_text)

                bot_text = _clean_bot_text(raw_text)
                _, routing_decision = _extract_routing_tag(raw_text)
                bot_text = fix_markdown_formatting(bot_text)

                print(f"==> ROUTING: {routing_decision}", flush=True)

                if calendar_action:
                    await ws.send_json({"type": "calendar_action"})
                if web_match:
                    await ws.send_json({"type": "web_search_pending"})

                await ws.send_json({"type": "text_done", "text": bot_text})

                if not sentences_for_tts:
                    await ws.send_json({"type": "error", "message": "No response generated"})
                    await ws.send_json({"type": "done"})
                    if tts_task:
                        tts_task.cancel()
                    continue

            except Exception as e:
                await ws.send_json({"type": "error", "message": f"AI failed: {str(e)}"})
                if tts_task:
                    tts_task.cancel()
                continue

            # ----------------------------------------------------------
            # VISUAL AID
            # ----------------------------------------------------------
            visual_aid = None
            visual_task = None

            if routing_decision in ("DIAGRAM", "CODE", "MATH", "HTML"):
                async def _gen_visual():
                    va, d_in, d_out = await _generate_visual(user_id, user_text, history_lines, file_context, image_attachments)
                    return va, d_in, d_out
                visual_task = asyncio.create_task(_gen_visual())
                await ws.send_json({"type": "visual_aid_pending"})

            # ----------------------------------------------------------
            # WAIT FOR TTS
            # ----------------------------------------------------------
            if tts_task:
                try:
                    await asyncio.wait_for(tts_task, timeout=30)
                except asyncio.TimeoutError:
                    tts_task.cancel()
                    print("==> TTS timed out")
                except Exception as e:
                    print(f"==> TTS error: {e}", flush=True)

            # ----------------------------------------------------------
            # WAIT FOR VISUAL + SEND
            # ----------------------------------------------------------
            if visual_task:
                try:
                    visual_aid, d_in, d_out = await visual_task
                    diagram_input_tokens = d_in
                    diagram_output_tokens = d_out
                except Exception as e:
                    print(f"==> Visual task error: {e}", flush=True)

            await _send_and_save_visual(ws, visual_aid, routing_decision, chat_id)

            # ----------------------------------------------------------
            # EXECUTE CALENDAR ACTION — capture second pass text
            # ----------------------------------------------------------
            second_pass_text = None

            if calendar_action and calendar_payload:
                second_pass_text = await _execute_calendar_action(
                    ws=ws, user_id=user_id,
                    calendar_action=calendar_action, calendar_payload=calendar_payload,
                    default_integration=default_integration, user_text=user_text,
                    system_prompt=system_prompt, voice_id=voice_id, audio_on=audio_on,
                )

            if web_match:
                second_pass_text = await _stream_second_pass(
                    ws=ws, user_id=user_id, user_text=user_text,
                    context_text=get_web_search().web_search(web_match.group(1), 3),
                    system_prompt=system_prompt, voice_id=voice_id, audio_on=audio_on,
                )

            # ----------------------------------------------------------
            # SAVE TO DB — save first pass, then second pass if exists
            # ----------------------------------------------------------
            try:
                rag.add_message(chat_id=chat_id, role="assistant", content=bot_text)
                rag.update_last_message(chat_id=chat_id, last_message=bot_text)
            except Exception:
                pass

            if second_pass_text:
                try:
                    rag.add_message(chat_id=chat_id, role="assistant", content=second_pass_text)
                    rag.update_last_message(chat_id=chat_id, last_message=second_pass_text)
                except Exception:
                    pass

            await _update_summary(smgr, chat_id, history, user_text, second_pass_text or bot_text)

            # ----------------------------------------------------------
            # COST TRACKING
            # ----------------------------------------------------------
            try:
                model = rag.get_model(user_id)
                cost = account_manager.processUsedCost(
                    input_tokens=chat_input_tokens + diagram_input_tokens,
                    output_tokens=chat_output_tokens + diagram_output_tokens,
                    SST_Length_seconds=len(audio_bytes) / 16000 if audio_bytes else 0,
                    webSearch=bool(web_match),
                    voice_on=bool(audio_on),
                    diagram_on=bool(visual_aid),
                    pro_mode=bool(pro_mode),
                    image_count=len(image_attachments),
                    model=model,
                )
                credits_used = cost / 0.15
                remaining = rag.deductCredits(user_id, credits_used)
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

    MIN_CREDITS_PER_TURN = 0.05
    print(f"==> embed WS opened for api_key={api_key}, owner={owner_user_id}")

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
            # LIMITS
            # ----------------------------------------------------------
            if key_data.get("conversations_used", 0) >= key_data.get("monthly_limit", 500):
                await ws.send_json({"type": "error", "message": "Monthly conversation limit reached", "code": "LIMIT_REACHED"})
                await ws.send_json({"type": "done"})
                continue

            try:
                current_credits = rag.getBusinessCredits(owner_user_id)
                if current_credits < MIN_CREDITS_PER_TURN:
                    await ws.send_json({"type": "error", "message": "This business has run out of credits.", "code": "NO_CREDITS"})
                    await ws.send_json({"type": "done"})
                    continue
            except Exception as e:
                print(f"==> Credit check failed: {e}")
                await ws.send_json({"type": "error", "message": "Temporary billing error.", "code": "BILLING_ERROR"})
                await ws.send_json({"type": "done"})
                continue

            try:
                daily_cost = rag.getApiKeyDailyCost(api_key)
                daily_cap = key_data.get("daily_cost_cap", 10.00)
                if daily_cost >= daily_cap:
                    await ws.send_json({"type": "error", "message": "Daily usage cap reached.", "code": "DAILY_CAP"})
                    await ws.send_json({"type": "done"})
                    continue
            except Exception:
                pass

            # ----------------------------------------------------------
            # EXTRACT CONFIG
            # ----------------------------------------------------------
            business_name        = key_data.get("business_name") or "this business"
            business_description = key_data.get("business_description") or ""
            assistant_name       = key_data.get("assistant_name") or "Assistant"
            version              = key_data.get("assistant_version", "professional")
            model                = rag.get_model(owner_user_id) or "gemini"
            diagrams_enabled     = rag.getEmbedDiagrams(owner_user_id)

            user_text  = _as_str(payload.get("message"))
            voice_id   = _as_str(payload.get("voice_name"))
            audio_on   = bool(payload.get("voice_on", True))
            raw_audio  = payload.get("audio_bytes")
            session_id = _as_str(payload.get("session_id")) or f"embed_{api_key}_{uuid.uuid4().hex[:12]}"

            rag.get_or_create_embed_session(session_id, api_key, owner_user_id)

            if raw_audio and "," in raw_audio:
                raw_audio = raw_audio.split(",", 1)[1]
            audio_bytes = base64.b64decode(raw_audio) if raw_audio else None

            embed_input_tokens = embed_output_tokens = 0
            diagram_input_tokens = diagram_output_tokens = 0
            second_pass_text = None
            web_match = None

            if not rag.hasEnoughCredits(owner_user_id):
                await ws.send_json({"type": "error", "message": "No credits remaining.", "code": "NO_CREDITS"})
                await ws.send_json({"type": "done"})
                continue

            # ----------------------------------------------------------
            # STT
            # ----------------------------------------------------------
            if audio_bytes:
                try:
                    user_text = get_stt().get_transcript(audio_bytes)
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
            # HISTORY
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
                if chunks and chunks[0].get("similarity", 0) >= 0.2:
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
            kb_section = rag_context if rag_context else "No specific documents loaded."

            system_prompt = (
                f"You are {assistant_name}, a support assistant for {business_name}. "
                f"{business_description}\n\n"
                "RULES\n\n"
                f"1. ONLY use information from the knowledge base below to answer questions about {business_name}. "
                "If the answer is not there, say so directly and suggest the user contact the business.\n\n"
                "2. Never invent, guess, or fill in gaps. If you are not sure, say I don't have that information.\n\n"
                "3. Never make up contact details, policies, pricing, hours, or product features.\n\n"
                f"4. Stay on topic. You help with {business_name} only. Politely redirect off-topic questions.\n\n"
                "5. Keep responses under 80 words. Use short sentences. This will be spoken aloud, not read on screen.\n\n"
                f"6. {tone} Use I statements. Sound like a helpful person, not a corporate script.\n\n"
                "7. No markdown, no formatting, no lists, no headers, no bold, no asterisks. Plain conversational sentences only.\n\n"
                "8. When you don't know something, always offer a next step: You could reach out to them directly for that.\n\n"
                "9. Never say based on my training, as an AI, or I believe. Just answer naturally or say you don't know.\n\n"
                f"10. If someone asks who you are, say: I'm {assistant_name}, a support assistant for {business_name}.\n\n"
                f"KNOWLEDGE BASE\n"
                f"Everything you know about {business_name} is below. If something is not here, you do not know it.\n\n"
                f"{kb_section}"
            )

            # Knowledge graph structured data
            system_prompt += _build_knowledge_graph_prompt(api_key)

            # Availability
            system_prompt += _build_availability_prompt(api_key)

            # Calendar integrations (on behalf of business owner)
            system_prompt += _build_integration_prompt(owner_user_id)
            default_integration = rag.getDefaultIntegration(owner_user_id)

            # Web search
            system_prompt += _build_websearch_prompt()

            # Visual aids
            if diagrams_enabled:
                system_prompt += (
                    "\n\nVISUAL AIDS: You can generate visual aids. End your response with one of these tags:\n"
                    "[NONE] - no visual needed\n"
                    "[DIAGRAM] - flowchart or architecture diagram\n"
                    "[CODE] - code snippet\n"
                    "[MATH] - equation or formula\n"
                    "[HTML] - interactive visual\n"
                    "Only use a visual tag if it genuinely helps. Default to [NONE]."
                )

            if audio_on:
                system_prompt += "\n\nLENGTH RULE: This response will be spoken aloud. Keep it under 80 words. Lead with the core answer."

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            system_prompt += f"\n\nToday's date is {today} (UTC). The user is in Melbourne, Australia (AEST, UTC+10). When booking calendar events, use Melbourne local time."

            # ----------------------------------------------------------
            # USER PROMPT WITH HISTORY
            # ----------------------------------------------------------
            history_lines = _build_history_lines(history[-6:])
            if history_lines:
                user_prompt = "Conversation history:\n" + "\n".join(history_lines) + f"\n\nLatest user message:\n{user_text}"
            else:
                user_prompt = user_text

            # ----------------------------------------------------------
            # GENERATE RESPONSE
            # ----------------------------------------------------------
            tts_queue = tts_task = tts_done_event = None

            try:
                if audio_on:
                    tts_queue = asyncio.Queue()
                    tts_done_event = asyncio.Event()
                    tts_task = asyncio.create_task(_tts_pipeline(ws, tts_queue, voice_id, tts_done_event))

                print(f"==> embed stream starting at {time.time() - t0:.2f}s", flush=True)

                full_text_parts, sentences_for_tts, embed_input_tokens, embed_output_tokens = await _stream_ai_response(
                    ws, api_key, system_prompt, user_prompt,
                    [], audio_on, voice_id, tts_queue, False
                )

                if tts_done_event:
                    tts_done_event.set()

                raw_text = "".join(full_text_parts).strip()
                print(f"==> embed FULL RAW: {repr(raw_text)}", flush=True)

                calendar_action, calendar_payload = _detect_calendar_action(raw_text)
                web_match = _detect_web_search(raw_text)

                _, routing_decision = _extract_routing_tag(raw_text)
                if not diagrams_enabled:
                    routing_decision = "NONE"

                bot_text = _clean_bot_text(raw_text)
                bot_text = fix_markdown_formatting(bot_text)

                if calendar_action:
                    await ws.send_json({"type": "calendar_action"})
                if web_match:
                    await ws.send_json({"type": "web_search_pending"})

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
            # VISUAL AID
            # ----------------------------------------------------------
            visual_aid = None
            visual_task = None

            if diagrams_enabled and routing_decision in ("DIAGRAM", "CODE", "MATH", "HTML"):
                async def _gen_embed_visual():
                    va, d_in, d_out = await _generate_visual(api_key, user_text, history_lines, "", [])
                    return va, d_in, d_out
                visual_task = asyncio.create_task(_gen_embed_visual())
                await ws.send_json({"type": "visual_aid_pending"})

            # ----------------------------------------------------------
            # WAIT FOR TTS
            # ----------------------------------------------------------
            if tts_task:
                try:
                    await asyncio.wait_for(tts_task, timeout=30)
                except asyncio.TimeoutError:
                    tts_task.cancel()
                    print("==> embed TTS timed out")
                except Exception as e:
                    print(f"==> embed TTS error: {e}", flush=True)

            # ----------------------------------------------------------
            # WAIT FOR VISUAL + SEND
            # ----------------------------------------------------------
            if visual_task:
                try:
                    visual_aid, d_in, d_out = await visual_task
                    diagram_input_tokens = d_in
                    diagram_output_tokens = d_out
                except Exception as e:
                    print(f"==> embed visual task error: {e}", flush=True)

            if diagrams_enabled:
                await _send_and_save_visual(ws, visual_aid, routing_decision, session_id)
            else:
                try:
                    await ws.send_json({"type": "visual_aid_none"})
                except Exception:
                    pass

            # ----------------------------------------------------------
            # CALENDAR ACTION
            # ----------------------------------------------------------
            if calendar_action and calendar_payload:
                second_pass_text = await _execute_calendar_action(
                    ws=ws, user_id=owner_user_id,
                    calendar_action=calendar_action, calendar_payload=calendar_payload,
                    default_integration=default_integration, user_text=user_text,
                    system_prompt=system_prompt, voice_id=voice_id, audio_on=audio_on,
                )

            # ----------------------------------------------------------
            # WEB SEARCH
            # ----------------------------------------------------------
            if web_match:
                second_pass_text = await _stream_second_pass(
                    ws=ws, user_id=owner_user_id, user_text=user_text,
                    context_text=get_web_search().web_search(web_match.group(1), 3),
                    system_prompt=system_prompt, voice_id=voice_id, audio_on=audio_on,
                )

            # ----------------------------------------------------------
            # SAVE TO DB
            # ----------------------------------------------------------
            try:
                rag.add_message(chat_id=session_id, role="assistant", content=bot_text)
                rag.update_last_message(chat_id=session_id, last_message=bot_text)
            except Exception as e:
                print(f"==> Failed to save assistant message: {e}")

            if second_pass_text:
                try:
                    rag.add_message(chat_id=session_id, role="assistant", content=second_pass_text)
                    rag.update_last_message(chat_id=session_id, last_message=second_pass_text)
                except Exception:
                    pass

            # ----------------------------------------------------------
            # BILLING
            # ----------------------------------------------------------
            try:
                rag.incrementConversationCount(api_key)
            except Exception as e:
                print(f"==> Failed to increment conversation count: {e}")

            try:
                cost_aud = account_manager.processUsedCost(
                    input_tokens=embed_input_tokens + diagram_input_tokens,
                    output_tokens=embed_output_tokens + diagram_output_tokens,
                    outputText=bot_text,
                    SST_Length_seconds=len(audio_bytes) / 32000 if audio_bytes else 0,
                    webSearch=bool(web_match),
                    voice_on=bool(audio_on),
                    diagram_on=bool(visual_aid),
                    model=model,
                )
                try:
                    rag.addApiKeyCost(api_key, cost_aud)
                except Exception as e:
                    print(f"==> Failed to add api_key cost: {e}")

                try:
                    new_balance = rag.deductBusinessCredits(owner_user_id, cost_aud)
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