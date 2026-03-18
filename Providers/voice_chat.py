import os
import base64
import re
import httpx
from typing import List, Dict, Any, Tuple
from fastapi.concurrency import run_in_threadpool
from Providers.phenome_provider import TextVisemeProvider


# ---------------------------------------------------------------------------
# ARPAbet phoneme → Preston Blair viseme ID (0–8)
# ---------------------------------------------------------------------------
ARPABET_TO_VISEME = {
    "AA": 3, "AE": 3, "AH": 3, "AO": 3,
    "AW": 4, "AY": 3, "EH": 2, "ER": 2,
    "EY": 2, "IH": 2, "IY": 2, "OW": 4,
    "OY": 4, "UH": 4, "UW": 4,
    "B":  1, "CH": 7, "D":  8, "DH": 6,
    "F":  5, "G":  8, "HH": 8, "JH": 7,
    "K":  8, "L":  8, "M":  1, "N":  8,
    "NG": 8, "P":  1, "R":  8, "S":  8,
    "SH": 7, "T":  8, "TH": 6, "V":  5,
    "W":  4, "Y":  2, "Z":  8, "ZH": 7,
}


class VoiceChatSystem:
    def __init__(self):
        self.api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
        self.voice_id = (os.getenv("ELEVENLABS_VOICE_ID") or "").strip()

        if not self.api_key:
            raise RuntimeError("Missing ELEVENLABS_API_KEY env var")
        if not self.voice_id:
            raise RuntimeError("Missing ELEVENLABS_VOICE_ID env var")

        # Fallback viseme provider in case timestamp API fails
        self.viseme_provider = TextVisemeProvider()

    async def synthesize_mp3_with_visemes(
        self,
        text: str,
        voice_name: str = None,
    ) -> Tuple[bytes, List[Dict[str, Any]]]:
        return await run_in_threadpool(self._synthesize_blocking, text)

    def _synthesize_blocking(self, text: str) -> Tuple[bytes, List[Dict[str, Any]]]:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/with-timestamps"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "output_format": "mp3_44100_128",
        }

        response = httpx.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Decode audio
        audio_bytes = base64.b64decode(data["audio_base64"])

        # Extract alignment
        alignment = data.get("alignment", {})
        characters = alignment.get("characters", [])
        start_times = alignment.get("character_start_times_seconds", [])
        end_times = alignment.get("character_end_times_seconds", [])

        visemes = self._build_visemes(characters, start_times, end_times)

        return audio_bytes, visemes

    def _build_visemes(
        self,
        characters: List[str],
        start_times: List[float],
        end_times: List[float],
    ) -> List[Dict[str, Any]]:
        import nltk
        from nltk.corpus import cmudict
        nltk.download("cmudict", quiet=True)
        cmu = cmudict.dict()

        word_windows = self._get_word_windows(characters, start_times, end_times)
        visemes = []

        for word, word_start, word_end in word_windows:
            clean = re.sub(r"[^a-z']", "", word.lower())
            if not clean:
                continue

            phonemes = self._word_to_phonemes(clean, cmu)
            if not phonemes:
                visemes.append({"t_ms": int(word_start * 1000), "viseme_id": 0})
                continue

            word_duration = word_end - word_start
            phoneme_duration = word_duration / len(phonemes)

            for i, phoneme in enumerate(phonemes):
                t_ms = int((word_start + i * phoneme_duration) * 1000)
                viseme_id = ARPABET_TO_VISEME.get(phoneme, 8)
                visemes.append({"t_ms": t_ms, "viseme_id": viseme_id})

        if end_times:
            visemes.append({"t_ms": int(end_times[-1] * 1000), "viseme_id": 0})

        return visemes

    def _get_word_windows(
        self,
        characters: List[str],
        start_times: List[float],
        end_times: List[float],
    ) -> List[Tuple[str, float, float]]:
        words = []
        current_word = ""
        word_start = None

        for char, start, end in zip(characters, start_times, end_times):
            if char in (" ", "\n"):
                if current_word:
                    words.append((current_word, word_start, end))
                    current_word = ""
                    word_start = None
            else:
                if word_start is None:
                    word_start = start
                current_word += char

        if current_word and word_start is not None:
            words.append((current_word, word_start, end_times[-1]))

        return words

    def _word_to_phonemes(self, word: str, cmu: dict) -> List[str]:
        entries = cmu.get(word)
        if not entries:
            return []
        return [re.sub(r"\d", "", p) for p in entries[0]]