import os
import json
import firebase_admin
from firebase_admin import credentials, auth
from fastapi import HTTPException, Security, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ---------------------------------------------------------------------------
# Initialise Firebase Admin SDK once at module level
# ---------------------------------------------------------------------------
_firebase_app = None

def _get_firebase_app():
    global _firebase_app
    if _firebase_app is None:
        service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        if not service_account_json:
            raise RuntimeError("FIREBASE_SERVICE_ACCOUNT env var is not set")
        service_account_dict = json.loads(service_account_json)
        cred = credentials.Certificate(service_account_dict)
        _firebase_app = firebase_admin.initialize_app(cred)
    return _firebase_app


async def verify_token_from_string(token: str) -> dict:
    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception as e:
        print(f"==> Token verify failed: {e}")
        raise

# ---------------------------------------------------------------------------
# HTTP endpoint auth — use as a FastAPI dependency
# ---------------------------------------------------------------------------
_bearer = HTTPBearer()

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> dict:
    """
    FastAPI dependency for HTTP endpoints.
    Usage: @router.post("/something")
           async def my_endpoint(user=Depends(verify_token)):
    """
    _get_firebase_app()
    token = credentials.credentials
    try:
        decoded = auth.verify_id_token(token)
        return decoded  # contains uid, email, etc.
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")


# ---------------------------------------------------------------------------
# WebSocket auth — call manually since WebSockets can't use Depends easily
# ---------------------------------------------------------------------------
async def verify_ws_token(ws: WebSocket) -> dict:
    """
    Call this at the start of your WebSocket handler.
    Expects the token in the query param: ws://...?token=<firebase_id_token>
    """
    _get_firebase_app()
    token = ws.query_params.get("token")
    if not token:
        await ws.close(code=4001, reason="Missing auth token")
        raise ValueError("Missing token")
    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception as e:
        await ws.close(code=4001, reason="Invalid token")
        raise ValueError(f"Invalid token: {str(e)}")