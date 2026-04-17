import os
import httpx
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID     = os.getenv("MICROSOFT_CLIENT_ID")
CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET")
TENANT_ID     = os.getenv("MICROSOFT_TENANT_ID")
REDIRECT_URI  = os.getenv("MICROSOFT_REDIRECT_URI")  # e.g. https://yourdomain.com/auth/microsoft/callback

SCOPES = [
    "offline_access",
    "User.Read",
    "Calendars.ReadWrite",
]

AUTHORITY = "https://login.microsoftonline.com/common"
TOKEN_URL  = f"{AUTHORITY}/oauth2/v2.0/token"
AUTH_URL   = f"{AUTHORITY}/oauth2/v2.0/authorize"


def build_auth_url(state: str = "") -> str:
    """Builds the Microsoft login URL to redirect the user to."""

    scope_str = " ".join(SCOPES)
    return (
        f"{AUTH_URL}"
        f"?client_id={CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_mode=query"
        f"&scope={scope_str}"
        f"&state={state}"
    )


async def exchange_code_for_tokens(code: str) -> dict:
    """
    Exchanges the auth code Microsoft gave us for access + refresh tokens.
    Returns the raw token response as a dict.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(TOKEN_URL, data={
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code":          code,
            "redirect_uri":  REDIRECT_URI,
            "grant_type":    "authorization_code",
        })
        response.raise_for_status()
        return response.json()


async def refresh_access_token(refresh_token: str) -> dict:
    """
    Uses the refresh token to get a new access token silently.
    Call this when the stored access token is expired.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(TOKEN_URL, data={
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type":    "refresh_token",
            "scope":         " ".join(SCOPES),
        })
        response.raise_for_status()
        return response.json()


def token_expiry_from_response(token_response: dict) -> datetime:
    """Converts the expires_in seconds from Microsoft into an absolute UTC datetime."""
    expires_in = int(token_response.get("expires_in", 3600))
    return datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)  # 60s buffer


async def get_valid_access_token(user_id: str, rag) -> str | None:
    """
    Main helper your tools call. Checks if the stored token is valid,
    refreshes if needed, returns a ready-to-use access token.
    Returns None if the user hasn't connected Microsoft.
    """
    tokens = rag.get_microsoft_tokens(user_id)
    if not tokens:
        return None

    expires_at = tokens["token_expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)

    now = datetime.now(timezone.utc)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if now >= expires_at:
        # Token expired — refresh it
        try:
            new_tokens = await refresh_access_token(tokens["refresh_token"])
            new_expiry = token_expiry_from_response(new_tokens)
            rag.save_microsoft_tokens(
                user_id=user_id,
                access_token=new_tokens["access_token"],
                refresh_token=new_tokens.get("refresh_token", tokens["refresh_token"]),
                expires_at=new_expiry,
                scopes=tokens.get("scopes"),
            )
            return new_tokens["access_token"]
        except Exception as e:
            print(f"==> Microsoft token refresh failed for {user_id}: {e}")
            return None

    return tokens["access_token"]


async def list_calendar_events(access_token: str, days_ahead: int = 7) -> list[dict]:
    """Get upcoming events from the user's calendar."""
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days_ahead)

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://graph.microsoft.com/v1.0/me/events",
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "$select": "subject,start,end,location,organizer",
                "$filter": f"start/dateTime ge '{now.isoformat()}' and start/dateTime le '{end.isoformat()}'",
                "$orderby": "start/dateTime",
                "$top": "10",
            }
        )
        resp.raise_for_status()
        return resp.json().get("value", [])


async def create_calendar_event(
    access_token: str,
    subject: str,
    start: datetime,
    end: datetime,
    body: str = "",
    attendee_emails: list[str] = [],
) -> dict:
    """Book a new calendar event."""

    import base64, json
    
    def decode_jwt_payload(token):
        try:
            payload = token.split(".")[1]
            payload += "=" * (4 - len(payload) % 4)
            return json.loads(base64.b64decode(payload))
        except:
            return {}
    
    decoded = decode_jwt_payload(access_token)
    print(f"==> TOKEN AUDIENCE: {decoded.get('aud')}")
    print(f"==> TOKEN SCOPES: {decoded.get('scp')}")
    print(f"==> TOKEN UPN: {decoded.get('upn')}")
    payload = {
        "subject": subject,
        "body": {"contentType": "Text", "content": body},
        "start": {"dateTime": start.isoformat(), "timeZone": "UTC"},
        "end":   {"dateTime": end.isoformat(),   "timeZone": "UTC"},
        "attendees": [
            {"emailAddress": {"address": e}, "type": "required"}
            for e in attendee_emails
        ],
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://graph.microsoft.com/v1.0/me/events",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        print(f"==> Graph API status: {resp.status_code}")
        print(f"==> Graph API response: {resp.text}")
        resp.raise_for_status()
        return resp.json()