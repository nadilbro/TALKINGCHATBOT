from fastapi import APIRouter, Depends, Query
from fastapi.responses import RedirectResponse, JSONResponse
from Providers.firebase_auth import verify_token
from Providers.microsoft_auth import (
    build_auth_url,
    exchange_code_for_tokens,
    token_expiry_from_response,
)
from SQL.SQLManager import VectorRAGService

router = APIRouter(prefix="/auth/microsoft", tags=["microsoft"])
rag = VectorRAGService()


@router.get("/connect")
async def microsoft_connect(user=Depends(verify_token)):
    """
    Step 1: User hits this to start connecting their Microsoft account.
    We redirect them to Microsoft's login page.
    """
    user_id = user["uid"]
    auth_url = build_auth_url(state=user_id)  # Pass user_id as state so we know who this is after redirect
    return RedirectResponse(url=auth_url)


@router.get("/callback")
async def microsoft_callback(
    code: str = Query(None),
    state: str = Query(None),
    error: str = Query(None),
    error_description: str = Query(None),
):
    """
    Step 2: Microsoft redirects here after user approves access.
    We exchange the code for tokens and store them.
    """
    if error:
        print(f"==> Microsoft OAuth error: {error} — {error_description}")
        return RedirectResponse(url=f"/?microsoft_error={error}")

    if not code or not state:
        return JSONResponse(status_code=400, content={"error": "Missing code or state"})

    user_id = state  # We passed user_id as state in step 1

    try:
        token_response = await exchange_code_for_tokens(code)
        expires_at = token_expiry_from_response(token_response)

        rag.save_microsoft_tokens(
            user_id=user_id,
            access_token=token_response["access_token"],
            refresh_token=token_response["refresh_token"],
            expires_at=expires_at,
            scopes=token_response.get("scope"),
        )

        print(f"==> Microsoft connected for user {user_id}")
        return RedirectResponse(url="/?microsoft_connected=true")

    except Exception as e:
        print(f"==> Microsoft token exchange failed: {e}")
        return RedirectResponse(url="/?microsoft_error=token_exchange_failed")


@router.get("/status")
async def microsoft_status(user=Depends(verify_token)):
    """Check if a user has Microsoft connected."""
    user_id = user["uid"]
    tokens = rag.get_microsoft_tokens(user_id)
    return {"connected": tokens is not None}


@router.delete("/disconnect")
async def microsoft_disconnect(user=Depends(verify_token)):
    """Remove stored Microsoft tokens for this user."""
    user_id = user["uid"]
    rag.delete_microsoft_tokens(user_id)
    return {"disconnected": True}
