import time, hmac, hashlib, base64
from urllib.parse import quote

class Security:
    def b64url(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode().rstrip("=")

    def sign_avatar(self, worker_base: str, avatar_key: str, secret: str, ttl: int = 300) -> str:
        secret_clean = (secret or "").strip()
        worker_base = worker_base.rstrip("/")

        exp = int(time.time()) + ttl
        msg = f"{avatar_key}:{exp}".encode("utf-8")
        sig = hmac.new(secret_clean.encode("utf-8"), msg, hashlib.sha256).digest()

        print("SECRET_FPR:", hashlib.sha256(secret_clean.encode("utf-8")).hexdigest()[:12], flush=True)
        print("MSG:", f"{avatar_key}:{exp}", flush=True)
        print("SIG:", self.b64url(sig), flush=True)

        return f"{worker_base}/a/{quote(avatar_key)}?exp={exp}&sig={self.b64url(sig)}"
