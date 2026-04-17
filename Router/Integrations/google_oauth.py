from fastapi import APIRouter, Depends, Query
from fastapi.responses import RedirectResponse, JSONResponse
from Providers.firebase_auth import verify_token, verify_token_from_string
from Providers.Integrations.google_auth import (
    build_auth_url,
    exchange_code_for_tokens,
    token_expiry_from_response,
)
from SQL.SQLManager import VectorRAGService

router = APIRouter(prefix="/auth/google", tags=["google"])
rag = VectorRAGService()


@router.get("/connect")
async def google_connect(token: str = Query(...)):
    try:
        user = await verify_token_from_string(token)
    except Exception:
        return JSONResponse(status_code=401, content={"error": "Invalid token"})

    user_id = user["uid"]
    auth_url = build_auth_url(state=user_id)
    return RedirectResponse(url=auth_url)


@router.get("/callback")
async def google_callback(
    code: str = Query(None),
    state: str = Query(None),
    error: str = Query(None),
):
    if error:
        print(f"==> Google OAuth error: {error}")
        return RedirectResponse(url=f"/?google_error={error}")

    if not code or not state:
        return JSONResponse(status_code=400, content={"error": "Missing code or state"})

    user_id = state

    try:
        token_response = await exchange_code_for_tokens(code)
        expires_at = token_expiry_from_response(token_response)

        rag.save_google_tokens(
            user_id=user_id,
            access_token=token_response["access_token"],
            refresh_token=token_response["refresh_token"],
            expires_at=expires_at,
            scopes=token_response.get("scope"),
        )
        rag.set_integrator_active(user_id, True)
        if not rag.getDefaultIntegration(user_id):
            rag.set_default_integration(user_id, "google")
        print(f"==> Google connected for user {user_id}")
        return RedirectResponse(url="/?google_connected=true")

    except Exception as e:
        print(f"==> Google token exchange failed: {e}")
        return RedirectResponse(url="/?google_error=token_exchange_failed")


@router.get("/status")
async def google_status(user=Depends(verify_token)):
    user_id = user["uid"]
    tokens = rag.get_google_tokens(user_id)
    return {"connected": tokens is not None}


@router.delete("/disconnect")
async def google_disconnect(user=Depends(verify_token)):
    user_id = user["uid"]
    rag.delete_google_tokens(user_id)
    
    ms_tokens = rag.get_microsoft_tokens(user_id)
    if not ms_tokens:
        rag.set_integrator_active(user_id, False)
        rag.set_default_integration(user_id, None)
    else:
        rag.set_default_integration(user_id, "microsoft")
    
    return {"disconnected": True}