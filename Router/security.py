from fastapi import Request, HTTPException
from urllib.parse import urlparse

def get_request_host(request: Request) -> str | None:
    origin = request.headers.get("origin")
    if origin:
        return urlparse(origin).hostname

    referer = request.headers.get("referer")
    if referer:
        return urlparse(referer).hostname

    return None

def host_allowed(host: str | None, allowed: list[str]) -> bool:
    if not host:
        return False
    host = host.lower()
    allowed = [d.lower() for d in allowed]
    return host in allowed
