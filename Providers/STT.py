from deepgram import DeepgramClient
import os
from typing import List, Dict, Any, Tuple

class DeepgramProvider:

    def __init__(self):
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")
        self.deepgram = DeepgramClient(api_key=api_key)


    def get_transcript(self, audio_bytes) -> str:
        try:

            response = self.deepgram.listen.rest.v("1").transcribe_file(
                {"buffer": audio_bytes, "mimetype": "audio/webm"},
                model="nova-3",
                language="en",
                smart_format=True,
            )
            transcript = response.results.channels[0].alternatives[0].transcript
            return transcript
        
        except Exception as e:
            print(f"Exception: {e}")