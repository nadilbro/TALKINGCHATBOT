import os
import base64
import re
import asyncio
from typing import List, Dict, Any, Tuple, Optional

import httpx

# ---------------------------------------------------------------------------
# CMU pronouncing dictionary (loaded once at module level)
# ---------------------------------------------------------------------------
import nltk
from nltk.corpus import cmudict
nltk.download("cmudict", quiet=True)
_CMU = cmudict.dict()


# ---------------------------------------------------------------------------
# ARPAbet phoneme → Azure-style viseme ID (0–21)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Relative duration weights per phoneme. Used to allocate each phoneme a
# proportional slice of the word's characters (and therefore of the word's
# real audio time). Stops are short, vowels are long, diphthongs longest.
# Values are relative, not milliseconds — only ratios matter.
# ---------------------------------------------------------------------------
PHONEME_WEIGHTS = {
    # stops — brief, percussive
    "B": 0.50, "P": 0.50, "D": 0.50, "T": 0.45, "G": 0.50, "K": 0.50,
    # affricates
    "CH": 0.70, "JH": 0.70,
    # fricatives
    "F": 0.70, "V": 0.70, "S": 0.80, "Z": 0.80,
    "SH": 0.80, "ZH": 0.80, "TH": 0.70, "DH": 0.60, "HH": 0.50,
    # nasals
    "M": 0.70, "N": 0.65, "NG": 0.70,
    # liquids / glides
    "L": 0.70, "R": 0.70, "W": 0.70, "Y": 0.60,
    # monophthong vowels
    "AA": 1.20, "AE": 1.10, "AH": 0.90, "AO": 1.20, "EH": 1.00,
    "ER": 1.10, "IH": 0.90, "IY": 1.10, "UH": 0.90, "UW": 1.10,
    # diphthongs — two shapes, longest sounds in English
    "AY": 1.60, "AW": 1.60, "EY": 1.40, "OW": 1.40, "OY": 1.60,
}
_DEFAULT_WEIGHT = 0.8

# CMU stress digit → duration multiplier. Stressed vowels are audibly longer.
STRESS_MULT = {"0": 0.85, "1": 1.25, "2": 1.05}

# ---------------------------------------------------------------------------
# Diphthong splitting. A diphthong is two mouth shapes, not one — "eye" is an
# open "ah" gliding into a narrow "ee". Emitting a single static viseme for
# the whole sound is why long vowels look frozen. When enabled, each diphthong
# emits its START shape at phoneme onset and its END shape partway through.
#   value = (start_viseme, end_viseme, fraction_through_phoneme_for_end_shape)
# ---------------------------------------------------------------------------
SPLIT_DIPHTHONGS = True
DIPHTHONG_SPLIT = {
    "AY": (2, 6, 0.55),   # "eye":  aa → iy
    "EY": (4, 6, 0.55),   # "day":  eh → iy
    "OY": (8, 6, 0.55),   # "boy":  ow → iy
    "AW": (2, 7, 0.55),   # "how":  aa → uw
    "OW": (8, 7, 0.60),   # "go":   ow → uw (subtle)
}

# ---------------------------------------------------------------------------
# Custom pronunciations for words CMU doesn't know — brand names, product
# names, local terms. ARPAbet with stress digits. Extend freely.
# ---------------------------------------------------------------------------
CUSTOM_PRONUNCIATIONS: Dict[str, List[str]] = {
    "kannai":      ["K", "AE1", "N", "AY1"],
    "bubbleworks": ["B", "AH1", "B", "AH0", "L", "W", "ER1", "K", "S"],
    "chatabit":    ["CH", "AE1", "T", "AH0", "B", "IH0", "T"],
}

# ---------------------------------------------------------------------------
# Letter-level fallback for words not in CMU and not in the custom dict
# (names, slang, codes). Maps each letter to a plausible viseme and uses the
# letter's own ElevenLabs timestamp — so unknown words still get real mouth
# motion instead of one frozen neutral shape.
# ---------------------------------------------------------------------------
LETTER_TO_VISEME = {
    "a": 1, "e": 4, "i": 6, "o": 8, "u": 7,
    "b": 21, "p": 21, "m": 21,
    "f": 18, "v": 18,
    "w": 7, "q": 20,
    "l": 14, "r": 13,
    "s": 15, "z": 15, "c": 15, "x": 15,
    "t": 19, "d": 19, "n": 19,
    "k": 20, "g": 20,
    "h": 12, "j": 16, "y": 6,
    # digits — neutral talking shape; better than nothing for "2pm" etc.
    "0": 4, "1": 4, "2": 4, "3": 4, "4": 4,
    "5": 4, "6": 4, "7": 4, "8": 4, "9": 4,
}

# ---------------------------------------------------------------------------
# Stream shaping
# ---------------------------------------------------------------------------
# If the silent gap between two words exceeds this, insert a rest (viseme 0)
# so the mouth closes during commas/pauses instead of holding the last shape.
PAUSE_GAP_MS = 140

# Events closer together than this are visually indistinguishable.
_MIN_VISEME_GAP_MS = 25

# In a timing collision, these shapes win (a missed lip closure is the single
# most visible lipsync error; a missed neutral vowel is invisible).
_PRIORITY_VISEMES = {21, 18, 19}

# Closures/percussives re-fire when repeated ("did" needs its second D).
_VISEME_KEEP_REPEAT = {21, 19, 18, 17, 15}


class VoiceChatSystem:
    def __init__(self):
        self.api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("Missing ELEVENLABS_API_KEY env var")
        # Shared client: connection pooling across sentences removes a TLS
        # handshake (~100-200ms) from every synthesis call after the first.
        self._client = httpx.AsyncClient(timeout=30.0)

    async def aclose(self):
        await self._client.aclose()

    # ======================================================================
    # Public API — unchanged signature
    # ======================================================================
    async def synthesize_sentence(
        self,
        text: str,
        voice_id: str,
    ) -> Tuple[bytes, List[Dict[str, Any]], float]:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_flash_v2_5",
            "output_format": "mp3_44100_128",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speed": 1.0,
            },
        }

        response = await self._post_with_retry(url, headers, payload)
        data = response.json()

        audio_bytes = base64.b64decode(data["audio_base64"])
        alignment = data.get("alignment", {}) or {}
        characters: List[str] = alignment.get("characters", [])
        start_times: List[float] = alignment.get("character_start_times_seconds", [])
        end_times: List[float] = alignment.get("character_end_times_seconds", [])

        visemes = self._build_visemes(characters, start_times, end_times)
        duration = end_times[-1] if end_times else 0.0
        return audio_bytes, visemes, duration

    # ----------------------------------------------------------------------
    async def _post_with_retry(self, url, headers, payload, retries: int = 2):
        """Retry transient failures (rate limit / 5xx / transport hiccups)."""
        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = await self._client.post(url, headers=headers, json=payload)
                if resp.status_code in (429, 500, 502, 503) and attempt < retries:
                    await asyncio.sleep(0.4 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp
            except httpx.TransportError as e:
                last_exc = e
                if attempt < retries:
                    await asyncio.sleep(0.4 * (attempt + 1))
                    continue
                raise
        raise last_exc or RuntimeError("ElevenLabs request failed")

    # ======================================================================
    # Viseme construction
    # ======================================================================
    def _build_visemes(
        self,
        characters: List[str],
        start_times: List[float],
        end_times: List[float],
    ) -> List[Dict[str, Any]]:
        """
        Per word:
          1. Locate its character window in the ElevenLabs alignment.
          2. Phonemize (custom dict → CMU → letter-heuristic fallback).
          3. Allocate each phoneme a slice of the word's characters,
             PROPORTIONAL to its duration weight (stress-adjusted) — so a
             stressed vowel claims more real audio time than a T.
          4. Emit each viseme at the actual timestamp of its first character.
          5. Split diphthongs into start→end shape pairs.
        Plus: rest insertion during inter-word pauses, trailing rest at audio
        end, and priority-aware smoothing.
        """
        if not characters or not start_times or not end_times:
            return []

        word_windows = self._get_word_windows(characters)
        visemes: List[Dict[str, Any]] = []
        prev_word_end_s: Optional[float] = None

        for word, char_offset, char_end_idx in word_windows:
            clean = re.sub(r"[^a-z0-9']", "", word.lower())
            if not clean:
                continue

            last_char = min(char_end_idx - 1, len(end_times) - 1)
            word_start_s = start_times[char_offset]
            word_end_s = end_times[last_char]

            # ── Rest during pauses ────────────────────────────────────
            if (
                prev_word_end_s is not None
                and (word_start_s - prev_word_end_s) * 1000 > PAUSE_GAP_MS
            ):
                visemes.append({
                    "t_ms": int(prev_word_end_s * 1000),
                    "viseme_id": 0,
                })
            prev_word_end_s = word_end_s

            phonemes = self._word_to_phonemes(clean)

            # ── Fallback: letter-heuristic with real char timing ──────
            if not phonemes:
                visemes.extend(
                    self._letter_fallback(word, char_offset, char_end_idx, start_times)
                )
                continue

            visemes.extend(
                self._phonemes_to_visemes(
                    phonemes, char_offset, char_end_idx, start_times, end_times
                )
            )

        # Return to rest at audio end
        visemes.append({"t_ms": int(end_times[-1] * 1000), "viseme_id": 0})

        return self._smooth_visemes(visemes)

    # ----------------------------------------------------------------------
    def _phonemes_to_visemes(
        self,
        phonemes: List[str],          # WITH stress digits, e.g. ["HH", "AH0", "L", "OW1"]
        char_offset: int,
        char_end_idx: int,
        start_times: List[float],
        end_times: List[float],
    ) -> List[Dict[str, Any]]:
        n_chars = char_end_idx - char_offset
        if n_chars <= 0:
            return []

        # Stress-adjusted duration weight per phoneme
        bases: List[str] = []
        weights: List[float] = []
        for p in phonemes:
            base = re.sub(r"\d", "", p)
            stress = p[-1] if p and p[-1].isdigit() else None
            w = PHONEME_WEIGHTS.get(base, _DEFAULT_WEIGHT)
            if stress is not None:
                w *= STRESS_MULT.get(stress, 1.0)
            bases.append(base)
            weights.append(w)

        total_w = sum(weights) or 1.0
        out: List[Dict[str, Any]] = []
        cum = 0.0

        for base, w in zip(bases, weights):
            frac_start = cum / total_w
            cum += w
            frac_end = cum / total_w

            # Fractional interpolation WITHIN characters (not snapping to char
            # starts). Snapping makes consecutive phonemes in short words land
            # on the same char index → same timestamp → collision-dropped
            # vowels ("my", "kannai"). Interpolating keeps every phoneme at a
            # unique, monotonically increasing time while still using the real
            # per-character timing ElevenLabs measured.
            t_start = self._char_frac_time(
                frac_start * n_chars, char_offset, n_chars, start_times, end_times
            )
            t_end = self._char_frac_time(
                frac_end * n_chars, char_offset, n_chars, start_times, end_times
            )
            if t_end < t_start:
                t_end = t_start

            if SPLIT_DIPHTHONGS and base in DIPHTHONG_SPLIT:
                v_a, v_b, frac = DIPHTHONG_SPLIT[base]
                out.append({"t_ms": int(t_start * 1000), "viseme_id": v_a})
                t_mid = t_start + frac * (t_end - t_start)
                out.append({"t_ms": int(t_mid * 1000), "viseme_id": v_b})
            else:
                out.append({
                    "t_ms": int(t_start * 1000),
                    "viseme_id": ARPABET_TO_VISEME.get(base, 0),
                })

        return out

    # ----------------------------------------------------------------------
    def _char_frac_time(
        self,
        f: float,
        char_offset: int,
        n_chars: int,
        start_times: List[float],
        end_times: List[float],
    ) -> float:
        """
        Map a fractional character position (0.0 → n_chars) within a word to
        an audio time, interpolating inside the character's measured window.
        f = 1.5 means "halfway through the word's second character".
        """
        f = min(max(f, 0.0), n_chars - 1e-6)
        i = int(f)
        frac = f - i
        gidx = min(char_offset + i, len(start_times) - 1)
        span = max(end_times[gidx] - start_times[gidx], 0.0)
        return start_times[gidx] + frac * span

    # ----------------------------------------------------------------------
    def _letter_fallback(
        self,
        word: str,
        char_offset: int,
        char_end_idx: int,
        start_times: List[float],
    ) -> List[Dict[str, Any]]:
        """
        Unknown word (name, brand, code): approximate mouth motion from its
        letters, each anchored to that letter's OWN timestamp. Far better
        than freezing one neutral shape across the whole word.
        """
        out: List[Dict[str, Any]] = []
        last_idx = len(start_times) - 1
        for i, ch in enumerate(word.lower()):
            vid = LETTER_TO_VISEME.get(ch)
            if vid is None:
                continue
            gidx = min(char_offset + i, last_idx)
            out.append({"t_ms": int(start_times[gidx] * 1000), "viseme_id": vid})
        if not out:
            gidx = min(char_offset, last_idx)
            out.append({"t_ms": int(start_times[gidx] * 1000), "viseme_id": 4})
        return out

    # ----------------------------------------------------------------------
    def _smooth_visemes(self, visemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        1. Sort (defensive).
        2. Collision handling (< _MIN_VISEME_GAP_MS apart): high-priority
           shapes (lip closures, F/V tuck, stops) REPLACE the colliding
           previous event instead of being dropped — a missed M closure is
           the most visible lipsync error there is.
        3. Smart dedup: identical consecutive IDs collapse unless the shape
           is a closure/percussive that needs to re-fire ("did", "mama").
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

            if gap < _MIN_VISEME_GAP_MS:
                if (
                    v["viseme_id"] in _PRIORITY_VISEMES
                    and prev["viseme_id"] not in _PRIORITY_VISEMES
                ):
                    # Keep the earlier slot but show the more critical shape
                    prev["viseme_id"] = v["viseme_id"]
                continue

            if v["viseme_id"] == prev["viseme_id"]:
                if v["viseme_id"] in _VISEME_KEEP_REPEAT:
                    smoothed.append(v)
                continue

            smoothed.append(v)

        return smoothed

    # ----------------------------------------------------------------------
    def _get_word_windows(self, characters: List[str]) -> List[Tuple[str, int, int]]:
        """(word, global_char_start_index, global_char_end_index_exclusive)"""
        words: List[Tuple[str, int, int]] = []
        current = ""
        start_idx: Optional[int] = None

        for i, ch in enumerate(characters):
            if ch in (" ", "\n", "\t"):
                if current and start_idx is not None:
                    words.append((current, start_idx, i))
                current = ""
                start_idx = None
            else:
                if start_idx is None:
                    start_idx = i
                current += ch

        if current and start_idx is not None:
            words.append((current, start_idx, len(characters)))
        return words

    # ----------------------------------------------------------------------
    def _word_to_phonemes(self, word: str) -> List[str]:
        """Custom dict → CMU. Returns phonemes WITH stress digits, [] if unknown."""
        custom = CUSTOM_PRONUNCIATIONS.get(word)
        if custom:
            return list(custom)
        entries = _CMU.get(word)
        if not entries:
            return []
        return list(entries[0])