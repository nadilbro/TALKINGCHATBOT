from fastapi import APIRouter, Query

from Providers.APIContracts import SessionBase, SessionCreate, SessionDelete
from SQL.RAG import VectorRAGService
from Providers.ai_provider import AIProvider
from Providers.startup_provider import StartUp

router = APIRouter(prefix="/startup", tags=["startup"])

rag = VectorRAGService()
ai = AIProvider(rag)
startup = StartUp()
from Providers.firebase_auth import verify_token
from fastapi import Depends

@router.get("/initialise_sessions")
#This is used to initialise the sessions so upon the user login, the old sessions can load.
async def initialise_sessions(user_id: str = Query(...), user=Depends(verify_token)):
    return rag.get_session_history(user_id)

@router.post("/create_session")
async def create_session_route(data: SessionCreate, user=Depends(verify_token)):
    chat_id = rag.create_session(
        user_id=data.user_id,
        title=data.title,
        avatar_name=data.avatar_name
    )
    return {"chat_id": chat_id}

@router.post("/delete_session")
async def delete_session_route(data: SessionDelete, user=Depends(verify_token)):
    success = rag.delete_session(user_id=data.user_id, chat_id=data.id)
    return {"success": success}
