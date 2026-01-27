from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
import time
from Providers.APIContracts import ChatRequest, StatusResponse, allowed_websites
from SQL.RAG import VectorRAGService
from Providers.ai_provider import AIProvider
import os
from fastapi import UploadFile, File, Form
import tempfile
import tempfile
from openai import AsyncOpenAI

router = APIRouter(prefix="/chat", tags=["chat"])

rag = VectorRAGService()
ai = AIProvider(rag)
oai = AsyncOpenAI() 

@router.post("/stream")
async def chat_stream(req: ChatRequest):
    print("Chat_stream called")
    async def event_gen():
        yield ": stream opened\n\n"
        yield "data: <p>Checking that for you…</p>\n\n"

        msg = (req.message or "").strip()
        if not msg:
            yield "data: <p>I didn’t catch that—what can I help with?</p>\n\n"
            yield "data: [DONE]\n\n"
            return
         
        # RAG
        (prompt, best_sim) = await rag.process_question(msg, req.site_id, 2)
        print("done Rag")
        # tenant description
        print("getting client")
        description = rag.get_client("description", req.site_id)
        print("Got client")
        system = f"""
        You are support for site_id="{req.site_id}".
        Business: {description}

        Rules:
        - Use ONLY CONTEXT. If missing, say you can't help and suggest what you can help with.
        - Do NOT repeat or rephrase the user's question.
        - Output exactly ONE <p>...</p> (no <h3>, no lists) unless the user asked for steps.
        - Max 100 words. HTML only (<p><br><b>).
        - Be friendly and make sure to add subheadings and headings to your answer using HTML

        CONTEXT:
        {prompt.context}
        """.strip()
        print("Starting Stream")
        t = time.time()
        print("Starting Stream")
        try:
            async for delta in ai.stream(site_id=req.site_id, system=system, user=msg):
                yield f"data: {delta}\n\n"
        except Exception as e:
            print("STREAM ERROR:", repr(e))
            yield f"data: <p><b>Error:</b> {repr(e)}</p>\n\n"
        finally:
            yield "data: [DONE]\n\n"
            print("Finished Stream")

        print("Streaming loop ended")
        print(time.time()-t)
    return StreamingResponse(event_gen(), media_type="text/event-stream")

@router.post("/listen")
async def listen(
    site_id: str = Form(...),
    audio: UploadFile = File(...)
        ):
    if audio.size > 2_000_000:  # ~2MB ≈ 2–3 minutes webm
        return ("Audio Too long")

    # Save uploaded audio to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp_path = tmp.name
        tmp.write(await audio.read())

    try:
        # Whisper transcription
        with open(tmp_path, "rb") as f:
            tr = await oai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
        text = getattr(tr, "text", None) 
        return {"site_id": site_id, "text": text}

    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
