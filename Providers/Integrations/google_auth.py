import os
import httpx
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI")

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid",
]

AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"


def build_auth_url(state: str = "") -> str:
    scope_str = " ".join(SCOPES)
    return (
        f"{AUTH_URL}"
        f"?client_id={CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_mode=query"
        f"&scope={scope_str}"
        f"&state={state}"
        f"&access_type=offline"
        f"&prompt=consent"
    )

async def delete_calendar_event(access_token: str, event_id: str) -> bool:
    async with httpx.AsyncClient() as client:
        resp = await client.delete(
            f"https://www.googleapis.com/calendar/v3/calendars/primary/events/{event_id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        return True
    
    
async def exchange_code_for_tokens(code: str) -> dict:
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
    async with httpx.AsyncClient() as client:
        response = await client.post(TOKEN_URL, data={
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type":    "refresh_token",
        })
        response.raise_for_status()
        return response.json()


def token_expiry_from_response(token_response: dict) -> datetime:
    expires_in = int(token_response.get("expires_in", 3600))
    return datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)


async def get_valid_access_token(user_id: str, rag) -> str | None:
    tokens = rag.get_google_tokens(user_id)
    if not tokens:
        return None

    expires_at = tokens["token_expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)

    now = datetime.now(timezone.utc)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if now >= expires_at:
        try:
            new_tokens = await refresh_access_token(tokens["refresh_token"])
            new_expiry = token_expiry_from_response(new_tokens)
            rag.save_google_tokens(
                user_id=user_id,
                access_token=new_tokens["access_token"],
                refresh_token=new_tokens.get("refresh_token", tokens["refresh_token"]),
                expires_at=new_expiry,
                scopes=tokens.get("scopes"),
            )
            return new_tokens["access_token"]
        except Exception as e:
            print(f"==> Google token refresh failed for {user_id}: {e}")
            return None

    return tokens["access_token"]


async def list_calendar_events(access_token: str, days_ahead: int = 7) -> list[dict]:
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days_ahead)

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "timeMin": now.isoformat(),
                "timeMax": end.isoformat(),
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": 10,
                "fields": "items(summary,start,end,location,organizer)",
            }
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        # Normalise to same shape as Microsoft response so the rest of the code works
        return [
            {
                "subject": e.get("summary", "Untitled"),
                "start":   {"dateTime": e.get("start", {}).get("dateTime") or e.get("start", {}).get("date")},
                "end":     {"dateTime": e.get("end", {}).get("dateTime") or e.get("end", {}).get("date")},
                "location": {"displayName": e.get("location", "")},
            }
            for e in items
        ]


async def create_calendar_event(
    access_token: str,
    subject: str,
    start: datetime,
    end: datetime,
    body: str = "",
    attendee_emails: list[str] = [],
) -> dict:
    payload = {
        "summary": subject,
        "description": body,
        "start": {
            "dateTime": start.isoformat(),
            "timeZone": "Australia/Melbourne",
        },
        "end": {
            "dateTime": end.isoformat(),
            "timeZone": "Australia/Melbourne",
        },
        "attendees": [{"email": e} for e in attendee_emails],
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        print(f"==> Google Calendar API status: {resp.status_code}")
        print(f"==> Google Calendar API response: {resp.text}")
        resp.raise_for_status()
        return resp.json()