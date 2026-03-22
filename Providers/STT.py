import httpx
import os

class DeepgramProvider:
    def __init__(self):
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")

    def get_transcript(self, audio_bytes, mimetype="audio/webm") -> str:
        try:
            response = httpx.post(
                "https://api.deepgram.com/v1/listen?model=nova-3&language=en&smart_format=true",
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": mimetype,
                },
                content=audio_bytes,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
            return transcript or ""
        except Exception as e:
            print(f"==> Deepgram exception: {e}")
            return ""
