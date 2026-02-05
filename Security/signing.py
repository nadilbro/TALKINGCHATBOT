import time, hmac, hashlib, base64
from urllib.parse import quote

class Security:

    def b64url(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode().rstrip("=")

    def sign_avatar(self, worker_base: str, avatar_key: str, secret: str, ttl: int = 300) -> str:
        # Clean + validate secret
        secret_clean = (secret or "").strip()
        if not secret_clean:
            raise RuntimeError("Avatar signing secret is empty")

        # Compute expiry FIRST
        exp = int(time.time()) + int(ttl)
        msg = f"{avatar_key}:{exp}".encode("utf-8")

        # Debug prints (now safe)
        print("SECRET_FPR:", hashlib.sha256(secret_clean.encode()).hexdigest()[:12], flush=True)
        print("MSG:", msg.decode(), flush=True)

        # Sign
        sig = hmac.new(
            secret_clean.encode("utf-8"),
            msg,
            hashlib.sha256
        ).digest()

        print("SIG:", self.b64url(sig), flush=True)

        worker_base = worker_base.rstrip("/")
        return f"{worker_base}/a/{quote(avatar_key)}?exp={exp}&sig={self.b64url(sig)}"
