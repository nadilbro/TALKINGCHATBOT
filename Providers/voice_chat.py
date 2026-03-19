import os
import re
import httpx
from typing import List, Dict, Any, Tuple, AsyncIterator
from Providers.phenome_provider import TextVisemeProvider


# ---------------------------------------------------------------------------
# ARPAbet phoneme → Azure-style viseme ID (0–21)
# ---------------------------------------------------------------------------
ARPABET_TO_VISEME = {
    "AA": 2,  "AE": 1,  "AH": 1,  "AO": 3,
    "AW": 9,  "AY": 11, "EH": 4,  "ER": 5,
    "EY": 11, "IH": 6,  "IY": 6,  "OW": 8,
    "OY": 10, "UH": 4,  "UW": 7,
    "B":  21, "CH": 16, "D":  19, "DH": 17,
    "F":  18, "G":  20, "HH": 12, "JH": 16,
    "K":  20, "L":  14, "M":  21, "N":  19,
    "NG": 20, "P":  21, "R":  13, "S":  15,
    "SH": 16, "T":  19, "TH": 17, "V":  18,
    "W":  7,  "Y":  6,  "Z":  15, "ZH": 16,
}


class VoiceChatSystem:
    def __init__(self):
        self.api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("Missing ELEVENLABS_API_KEY env var")
        self.viseme_provider = TextVisemeProvider()

    def get_visemes(self, text: str) -> List[Dict[str, Any]]:
        """
        Generate visemes instantly from text using NLTK.
        Called before/during streaming so visemes arrive fast.
        """
        return self.viseme_provider.get_visemes(text)

    async def stream_audio(self, text: str, voice_id: str) -> AsyncIterator[bytes]:
        """
        Streams audio chunks from ElevenLabs as they arrive.
        First chunk arrives in ~400ms.
        """
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "model_id": "eleven_flash_v2_5",
            "output_format": "mp3_44100_128",
        }

        with httpx.stream("POST", url, headers=headers, json=payload, timeout=30) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(chunk_size=32_000):
                if chunk:
                    yield chunk