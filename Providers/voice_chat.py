import os
import base64
import re
import httpx
import asyncio
from typing import List, Dict, Any, Tuple


# ---------------------------------------------------------------------------
# ARPAbet phoneme → Azure-style viseme ID (0–21)
# ---------------------------------------------------------------------------
# Adjustments from prior version:
#   - UH ("book") → 5 (slight pucker), was 4. UH is rounded, not neutral-open.
#   - Fallback for unknown phonemes is now 0 (silence) not 19 (D/T/N). Defaulting
#     to a distinct stop shape on unknown input causes visible flickering;
#     silence is unobtrusive.
ARPABET_TO_VISEME = {
    "AA": 2,  "AE": 1,  "AH": 1,  "AO": 3,
    "AW": 9,  "AY": 11, "EH": 4,  "ER": 5,
    "EY": 11, "IH": 6,  "IY": 6,  "OW": 8,
    "OY": 10, "UH": 5,  "UW": 7,
    "B":  21, "CH": 16, "D":  19, "DH": 17,
    "F":  18, "G":  20, "HH": 12, "JH": 16,
    "K":  20, "L":  14, "M":  21, "N":  19,
    "NG": 20, "P":  21, "R":  13, "S":  15,
    "SH": 16, "T":  19, "TH": 17, "V":  18,
    "W":  7,  "Y":  6,  "Z":  15, "ZH": 16,
}

# Visemes that benefit from a re-emphasis when repeated (vowels held longer).
# When two adjacent visemes share an ID and BOTH map to one of these, we keep
# the second one rather than deduping. Prevents "did" (D-IH-D) from losing its
# final closure or sustained vowels from getting a single keyframe.
_VISEME_KEEP_REPEAT = {21, 19, 18, 17, 15}  # closures and percussives

# Minimum gap between viseme events (ms). Below this, Rive can't visibly
# distinguish them and we'd just be flooding the state machine.
_MIN_VISEME_GAP_MS = 25


import nltk
from nltk.corpus import cmudict
nltk.download("cmudict", quiet=True)
_CMU = cmudict.dict()


class VoiceChatSystem:
    def __init__(self):
        self.api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("Missing ELEVENLABS_API_KEY env var")

    async def synthesize_sentence(
        self,
        text: str,
        voice_id: str,
    ) -> Tuple[bytes, List[Dict[str, Any]], float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._call_elevenlabs, text, voice_id
        )

    # ======================================================================
    # ElevenLabs
    # ======================================================================
    def _call_elevenlabs(
        self, text: str, voice_id: str
    ) -> Tuple[bytes, List[Dict[str, Any]], float]:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Changed: speed back to 1.0. The previous 0.9 was masking bad viseme
        # timing by stretching audio. Now that timing comes from actual char
        # timestamps, audio runs at normal pace and visemes track correctly.
        payload = {
            "text": text,
            "model_id": "eleven_flash_v2_5",
            "output_format": "mp3_44100_128",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speed": 1.0,
            }
        }

        response = httpx.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        audio_bytes = base64.b64decode(data["audio_base64"])

        alignment = data.get("alignment", {})
        characters: List[str] = alignment.get("characters", [])
        start_times: List[float] = alignment.get("character_start_times_seconds", [])
        end_times: List[float] = alignment.get("character_end_times_seconds", [])

        visemes = self._build_visemes(characters, start_times, end_times)
        duration = end_times[-1] if end_times else 0.0

        return audio_bytes, visemes, duration

    # ======================================================================
    # Viseme construction — character-anchored phoneme timing
    # ======================================================================
    def _build_visemes(
        self,
        characters: List[str],
        start_times: List[float],
        end_times: List[float],
    ) -> List[Dict[str, Any]]:
        """
        For each word in the audio:
          1. Get its character-level start/end window from ElevenLabs.
          2. Convert the word to ARPAbet phonemes via CMU dict.
          3. Distribute phonemes proportionally across the word's CHARACTERS,
             using the actual per-character timestamps (not even division).
          4. Emit a viseme event at the audio time of each phoneme's first char.

        This is dramatically more accurate than the old approach (even-split
        across word duration) because actual char durations vary — stops are
        short, vowels are long — and ElevenLabs measured them for us.
        """
        if not characters or not start_times or not end_times:
            return []

        word_windows = self._get_word_windows(characters, start_times, end_times)
        visemes: List[Dict[str, Any]] = []

        for word, char_offset, char_end_idx in word_windows:
            clean = re.sub(r"[^a-z']", "", word.lower())
            if not clean:
                continue

            phonemes = self._word_to_phonemes(clean)
            word_char_count = char_end_idx - char_offset

            # Fallback: unknown word → single neutral-open viseme at word start
            if not phonemes or word_char_count <= 0:
                visemes.append({
                    "t_ms": int(start_times[char_offset] * 1000),
                    "viseme_id": 4,  # neutral mid-open, generic "talking"
                })
                continue

            # Proportionally assign each phoneme a slice of the word's chars.
            # Each phoneme's t_ms = start_time of its first assigned char.
            n_phon = len(phonemes)
            for i, phoneme in enumerate(phonemes):
                # First char index for this phoneme, within the word.
                rel_char_idx = int(i * word_char_count / n_phon)
                global_char_idx = char_offset + rel_char_idx

                # Bounds safety — should never trigger, but cheap insurance
                if global_char_idx >= len(start_times):
                    global_char_idx = len(start_times) - 1

                t_ms = int(start_times[global_char_idx] * 1000)
                viseme_id = ARPABET_TO_VISEME.get(phoneme, 0)
                visemes.append({"t_ms": t_ms, "viseme_id": viseme_id})

        # Close with silence at audio end so the mouth returns to rest
        visemes.append({
            "t_ms": int(end_times[-1] * 1000),
            "viseme_id": 0,
        })

        return self._smooth_visemes(visemes)

    # ----------------------------------------------------------------------
    def _smooth_visemes(self, visemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Post-process the raw viseme stream:
          1. Sort by time (should already be sorted, but defensive).
          2. Drop events too close together to be visible (< _MIN_VISEME_GAP_MS).
          3. Smart dedup: collapse consecutive identical IDs UNLESS the ID is
             a closure/percussive (closures need to re-fire so e.g. "did" still
             gets its final D shape).
        """
        if not visemes:
            return visemes

        visemes.sort(key=lambda v: v["t_ms"])

        smoothed: List[Dict[str, Any]] = []
        for v in visemes:
            if not smoothed:
                smoothed.append(v)
                continue

            prev = smoothed[-1]
            gap = v["t_ms"] - prev["t_ms"]

            # Too close to matter visually → drop the later one, keeping the
            # earlier (more distinct) shape.
            if gap < _MIN_VISEME_GAP_MS:
                continue

            # Same shape as previous?
            if v["viseme_id"] == prev["viseme_id"]:
                # If it's a closure/percussive that needs re-firing (e.g. the
                # second D in "did"), keep it. Otherwise dedup.
                if v["viseme_id"] in _VISEME_KEEP_REPEAT:
                    smoothed.append(v)
                # else: silently dedup
                continue

            smoothed.append(v)

        return smoothed

    # ----------------------------------------------------------------------
    def _get_word_windows(
        self,
        characters: List[str],
        start_times: List[float],
        end_times: List[float],
    ) -> List[Tuple[str, int, int]]:
        """
        Returns: list of (word, char_start_index, char_end_index).
        char_*_index are positions in the global `characters` array, not times.
        Caller uses them to look up start_times[idx] / end_times[idx] directly.
        """
        words: List[Tuple[str, int, int]] = []
        current_word = ""
        word_start_idx: int | None = None

        for i, char in enumerate(characters):
            if char in (" ", "\n", "\t"):
                if current_word and word_start_idx is not None:
                    words.append((current_word, word_start_idx, i))
                    current_word = ""
                    word_start_idx = None
            else:
                if word_start_idx is None:
                    word_start_idx = i
                current_word += char

        # Trailing word with no whitespace after
        if current_word and word_start_idx is not None:
            words.append((current_word, word_start_idx, len(characters)))

        return words

    # ----------------------------------------------------------------------
    def _word_to_phonemes(self, word: str) -> List[str]:
        entries = _CMU.get(word)
        if not entries:
            return []
        # Strip stress digits (AH1 → AH). First pronunciation only — CMU often
        # lists multiple variants but the first is usually most common.
        return [re.sub(r"\d", "", p) for p in entries[0]]