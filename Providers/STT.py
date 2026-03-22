def get_transcript(self, audio_bytes, mimetype="audio/webm") -> str:
    try:
        print(f"==> Deepgram: sending {len(audio_bytes)} bytes", flush=True)
        response = httpx.post(
            "https://api.deepgram.com/v1/listen?model=nova-3&language=en&smart_format=true",
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": mimetype,
            },
            content=audio_bytes,
            timeout=30,
        )
        print(f"==> Deepgram status: {response.status_code}", flush=True)
        response.raise_for_status()
        data = response.json()
        print(f"==> Deepgram response: {data}", flush=True)
        transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        print(f"==> Transcript: '{transcript}'", flush=True)
        return transcript or ""
    except Exception as e:
        print(f"==> Deepgram exception: {e}", flush=True)
        return ""