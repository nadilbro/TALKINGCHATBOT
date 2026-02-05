from fastapi import APIRouter
from pydantic import BaseModel
import base64
import os
from fastapi.concurrency import run_in_threadpool

from Providers.APIContracts import VoiceChat
from Providers.ai_provider import AIProvider
from Providers.voice_chat import VoiceChatSystem
from Security.signing import Security
from fastapi import Request, HTTPException
from SQL.RAG import VectorRAGService


router = APIRouter(prefix="/voice", tags=["voice"])
rag = VectorRAGService()
ai = AIProvider(rag)
tts = VoiceChatSystem()
security = Security()

class VoiceInit(BaseModel):
    site_id: str

class VoiceRequest(BaseModel):
    site_id: str
    voice_name: str
    message: str

@router.post("/audio_chat_init")
async def audio_chat_init(details: VoiceInit):

    #First we get chatgpt reponse.
    #then we process audio
    # Unpack into two variables
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


@router.post("/audio_chat")
async def audio_chat(details: VoiceRequest):
    try: 
        (prompt, _) = await rag.process_question(details.message, details.site_id, 2)

        description = rag.get_client("description", details.site_id)

        print("Got client")
        system = f"""
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
        - If something is uncertain, say so briefly and naturally.

        Relevant context:
        {prompt.context}
        """.strip()

        # 1. AI
        text = await ai.chat(
            site_id=details.site_id,
            system=system,
            user=details.message
        )

    except Exception as e:
        print("AI ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=f"AI error: {e}")

    try:
        # 2. TTS
        audio_bytes, visemes = await run_in_threadpool(
            tts.synthesize,
            text,
            details.voice_name
        )

    except Exception as e:
        print("TTS ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")

    # 3. Encode safely
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "text": text,
        "audio": audio_b64,
        "visemes": visemes,
    }