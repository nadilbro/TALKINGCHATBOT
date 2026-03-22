from deepgram import DeepgramClient
import os
from typing import List, Dict, Any, Tuple

class DeepgramProvider:

    def __init__(self):
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")
        self.deepgram = DeepgramClient(api_key=api_key)


    def get_transcript(self, audio_bytes, mimetype="audio/webm") -> str:
        try:
            print(f"==> Deepgram: sending {len(audio_bytes)} bytes, mimetype={mimetype}")
            response = self.deepgram.listen.rest.v("1").transcribe_file(
                {"buffer": audio_bytes, "mimetype": mimetype},
                model="nova-3",
                language="en",
                smart_format=True,
            )
            print(f"==> Deepgram raw response: {response}")
            transcript = response.results.channels[0].alternatives[0].transcript
            return transcript or ""
        except Exception as e:
            print(f"==> Deepgram exception: {e}")
            return ""