from fastapi import APIRouter, Query

from Providers.APIContracts import SessionBase
from SQL.RAG import VectorRAGService
from Providers.ai_provider import AIProvider
from Providers.startup_provider import StartUp

router = APIRouter(prefix="/startup", tags=["startup"])

rag = VectorRAGService()
ai = AIProvider(rag)
startup = StartUp()


@router.get("/initialise_sessions")
#This is used to initialise the sessions so upon the user login, the old sessions can load.
async def initialise_sessions(user_id: str = Query(...)):
    return rag.get_session_history(user_id)

@router.post("/create_session")
#This is used to create a new session to be registered into the database
async def create_session_route(data: SessionBase):
    chat_id = rag.create_session(user_id=data.user_id, title=data.title)
    return {"chat_id": chat_id}

@router.post("/delete_session")
async def delete_session_route(data: SessionBase):
    success = rag.delete_session(user_id=data.user_id, chat_id=data.id)
    return {"success": success}
