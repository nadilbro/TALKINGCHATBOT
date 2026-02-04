import time, hmac, hashlib
import base64
from urllib.parse import quote


class Security():

    def b64url(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode().rstrip("=")

    def sign_avatar(self, worker_base: str, avatar_key: str, secret: str, ttl: int = 300) -> str:
        secret = (secret or "").strip()   # <-- add this
        worker_base = worker_base.rstrip("/")
        exp = int(time.time()) + ttl
        msg = f"{avatar_key}:{exp}".encode("utf-8")
        sig = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()
        return f"{worker_base}/a/{quote(avatar_key)}?exp={exp}&sig={self.b64url(sig)}"

