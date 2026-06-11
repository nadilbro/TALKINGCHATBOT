import os
import base64
import re
import asyncio
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional

import httpx

# ---------------------------------------------------------------------------
# CMU pronouncing dictionary — lazy download pattern avoids nltk's downloader
# index check on every cold boot (noticeable on Render).
# ---------------------------------------------------------------------------
import nltk
try:
    from nltk.corpus import cmudict
    _CMU = cmudict.dict()
except LookupError:
    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict
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

# Relative duration weights per phoneme (ratios only; stops short, vowels long)
PHONEME_WEIGHTS = {
    "B": 0.50, "P": 0.50, "D": 0.50, "T": 0.45, "G": 0.50, "K": 0.50,
    "CH": 0.70, "JH": 0.70,
    "F": 0.70, "V": 0.70, "S": 0.80, "Z": 0.80,
    "SH": 0.80, "ZH": 0.80, "TH": 0.70, "DH": 0.60, "HH": 0.50,
    "M": 0.70, "N": 0.65, "NG": 0.70,
    "L": 0.70, "R": 0.70, "W": 0.70, "Y": 0.60,
    "AA": 1.20, "AE": 1.10, "AH": 0.90, "AO": 1.20, "EH": 1.00,
    "ER": 1.10, "IH": 0.90, "IY": 1.10, "UH": 0.90, "UW": 1.10,
    "AY": 1.60, "AW": 1.60, "EY": 1.40, "OW": 1.40, "OY": 1.60,
}
_DEFAULT_WEIGHT = 0.8
STRESS_MULT = {"0": 0.85, "1": 1.25, "2": 1.05}

# ---------------------------------------------------------------------------
# Precomputed phoneme table: every "BASE" / "BASE0/1/2" string CMU can emit →
# (base, viseme_id, stress_adjusted_weight). Replaces per-phoneme regex +
# dict-chain lookups in the hot loop with a single dict hit.
# ---------------------------------------------------------------------------
_PHONEME_TABLE: Dict[str, Tuple[str, int, float]] = {}
for _base, _w in PHONEME_WEIGHTS.items():
    _vis = ARPABET_TO_VISEME.get(_base, 0)
    _PHONEME_TABLE[_base] = (_base, _vis, _w)
    for _s, _m in STRESS_MULT.items():
        _PHONEME_TABLE[_base + _s] = (_base, _vis, _w * _m)


def _phoneme_info(p: str) -> Tuple[str, int, float]:
    """(base, viseme_id, weight) for any ARPAbet token, stress-adjusted."""
    hit = _PHONEME_TABLE.get(p)
    if hit:
        return hit
    base = p.rstrip("0123456789")
    return (base, ARPABET_TO_VISEME.get(base, 0), _DEFAULT_WEIGHT)


# ---------------------------------------------------------------------------
# Diphthong splitting: emit start shape at onset, end shape partway through.
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
# Custom pronunciations (brand names etc). ARPAbet with stress digits.
# ---------------------------------------------------------------------------
CUSTOM_PRONUNCIATIONS: Dict[str, List[str]] = {
    "kannai":      ["K", "AE1", "N", "AY1"],
    "bubbleworks": ["B", "AH1", "B", "AH0", "L", "W", "ER1", "K", "S"],
    "chatabit":    ["CH", "AE1", "T", "AH0", "B", "IH0", "T"],
}

# ---------------------------------------------------------------------------
# Letter NAME phonemes — how TTS pronounces spelled-out letters ("pm" →
# "pee em"). Used for short vowel-less alpha runs CMU doesn't know (pm, tv,
# faq, lcd) so acronyms get the mouth shapes that are actually spoken.
# ---------------------------------------------------------------------------
LETTER_NAME_PHONEMES: Dict[str, List[str]] = {
    "a": ["EY1"], "b": ["B", "IY1"], "c": ["S", "IY1"], "d": ["D", "IY1"],
    "e": ["IY1"], "f": ["EH1", "F"], "g": ["JH", "IY1"], "h": ["EY1", "CH"],
    "i": ["AY1"], "j": ["JH", "EY1"], "k": ["K", "EY1"], "l": ["EH1", "L"],
    "m": ["EH1", "M"], "n": ["EH1", "N"], "o": ["OW1"], "p": ["P", "IY1"],
    "q": ["K", "Y", "UW1"], "r": ["AA1", "R"], "s": ["EH1", "S"],
    "t": ["T", "IY1"], "u": ["Y", "UW1"], "v": ["V", "IY1"],
    "w": ["D", "AH1", "B", "AH0", "L", "Y", "UW0"], "x": ["EH1", "K", "S"],
    "y": ["W", "AY1"], "z": ["Z", "IY1"],
}

# ---------------------------------------------------------------------------
# Number → spoken-word phonemes. TTS says "6am" as "six ay em" and "$15" as
# "fifteen dollars" — animating digits as frozen shapes looks dead. We expand
# digit runs to number words (all present in CMU) and phonemize those.
# ---------------------------------------------------------------------------
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _number_to_words(digits: str) -> List[str]:
    """'15' → ['fifteen']; '600' → ['six','hundred']; long runs → per digit."""
    if len(digits) > 6:
        return [_ONES[int(d)] for d in digits]
    n = int(digits)
    if n == 0:
        return ["zero"]

    def under_1000(x: int) -> List[str]:
        out: List[str] = []
        if x >= 100:
            out += [_ONES[x // 100], "hundred"]
            x %= 100
        if x >= 20:
            out.append(_TENS[x // 10])
            x %= 10
            if x:
                out.append(_ONES[x])
        elif x > 0:
            out.append(_ONES[x])
        return out

    words: List[str] = []
    if n >= 1000:
        words += under_1000(n // 1000) + ["thousand"]
        n %= 1000
    words += under_1000(n)
    return words


# ---------------------------------------------------------------------------
# Letter-level viseme fallback for tokens nothing else can phonemize.
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
    "0": 4, "1": 4, "2": 4, "3": 4, "4": 4,
    "5": 4, "6": 4, "7": 4, "8": 4, "9": 4,
}

# ---------------------------------------------------------------------------
# Stream shaping
# ---------------------------------------------------------------------------
PAUSE_GAP_MS = 140            # gap above this → insert rest (mouth closes)
_MIN_VISEME_GAP_MS = 25       # events closer than this are invisible
_PRIORITY_VISEMES = {21, 18, 19}        # closures win timing collisions
_VISEME_KEEP_REPEAT = {21, 19, 18, 17, 15}  # re-fire when repeated ("did")

_VOWELS = frozenset("aeiou")
_RUN_RE = re.compile(r"[a-z']+|[0-9]+")
_WHITESPACE = frozenset((" ", "\n", "\t", "\r"))


def _is_spoken_char(ch: str) -> bool:
    return ch.isalnum() or ch == "'" or ch == "\u2019"


# ---------------------------------------------------------------------------
# Token phonemization — cached. The same words ("the", "a", "is") repeat
# constantly; caching skips ALL per-token work on repeats.
# Returns (bases, viseme_ids, weights) tuples, or None → letter fallback.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8192)
def _token_phoneme_data(token: str) -> Optional[Tuple[Tuple[str, ...], Tuple[int, ...], Tuple[float, ...]]]:
    phonemes: List[str] = []

    # Whole-token custom/CMU first (handles apostrophe words like "don't")
    whole = CUSTOM_PRONUNCIATIONS.get(token) or (_CMU.get(token) or [None])[0]
    if whole:
        phonemes = list(whole)
    else:
        # Segment into alpha / digit runs ("2pm" → "2","pm"; "self-service"
        # → "self","service" since hyphen separates runs).
        runs = _RUN_RE.findall(token)
        if not runs:
            return None
        for run in runs:
            if run[0].isdigit():
                for word in _number_to_words(run):
                    entry = _CMU.get(word)
                    if entry:
                        phonemes.extend(entry[0])
                continue
            part = CUSTOM_PRONUNCIATIONS.get(run) or (_CMU.get(run) or [None])[0]
            if part:
                phonemes.extend(part)
            elif len(run) <= 2 or (len(run) <= 4 and not (_VOWELS & set(run))):
                # Short vowel-less run → spoken as spelled letters (pm, tv, lcd)
                for ch in run:
                    phonemes.extend(LETTER_NAME_PHONEMES.get(ch, []))
            else:
                return None  # unresolvable run → whole token to letter fallback

    if not phonemes:
        return None

    bases: List[str] = []
    visemes: List[int] = []
    weights: List[float] = []
    for p in phonemes:
        base, vis, w = _phoneme_info(p)
        bases.append(base)
        visemes.append(vis)
        weights.append(w)
    return tuple(bases), tuple(visemes), tuple(weights)


class VoiceChatSystem:
    def __init__(self):
        self.api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("Missing ELEVENLABS_API_KEY env var")
        # Shared client + pool: removes a TLS handshake (~100-200ms) from
        # every synthesis call after the first. connect=5s fails fast.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

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
          1. Locate its character window; TRIM unpronounced punctuation off
             both ends so the timing window covers only spoken characters
             (the comma in "hello," carries the pause time — including it
             stretched the last phoneme into the silence and hid the gap
             from the pause detector).
          2. Phonemize (custom → CMU → hyphen/number/acronym segmentation →
             letter fallback), cached per token.
          3. Allocate each phoneme a duration-weighted, stress-adjusted slice
             of the word's spoken characters; timestamp via fractional
             interpolation inside the real per-character timing.
          4. Split diphthongs into start→end shape pairs.
        Plus rest insertion during pauses, trailing rest, priority smoothing.
        """
        if not characters or not start_times or not end_times:
            return []

        n_times = len(start_times)
        word_windows = self._get_word_windows(characters)
        visemes: List[Dict[str, Any]] = []
        prev_word_end_s: Optional[float] = None

        for _word, char_offset, char_end_idx in word_windows:
            # ── Trim window to spoken chars only ──────────────────────
            ws, we = char_offset, min(char_end_idx, n_times)
            while ws < we and not _is_spoken_char(characters[ws]):
                ws += 1
            while we > ws and not _is_spoken_char(characters[we - 1]):
                we -= 1
            if ws >= we:
                continue  # pure punctuation token ("—", "...")

            token = "".join(characters[ws:we]).lower().replace("\u2019", "'")
            word_start_s = start_times[ws]
            word_end_s = end_times[we - 1]

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

            data = _token_phoneme_data(token)
            if data is None:
                visemes.extend(self._letter_fallback(token, ws, start_times))
                continue

            visemes.extend(
                self._phonemes_to_visemes(data, ws, we, start_times, end_times)
            )

        # Return to rest at audio end
        visemes.append({"t_ms": int(end_times[-1] * 1000), "viseme_id": 0})

        smoothed = self._smooth_visemes(visemes)

        # The min-gap filter can drop the terminal rest if the last shape
        # landed within 25ms of audio end — but the mouth MUST return to
        # rest after speech, so guarantee it.
        if smoothed and smoothed[-1]["viseme_id"] != 0:
            smoothed.append({"t_ms": int(end_times[-1] * 1000), "viseme_id": 0})
        return smoothed

    # ----------------------------------------------------------------------
    def _phonemes_to_visemes(
        self,
        data: Tuple[Tuple[str, ...], Tuple[int, ...], Tuple[float, ...]],
        char_offset: int,
        char_end_idx: int,
        start_times: List[float],
        end_times: List[float],
    ) -> List[Dict[str, Any]]:
        bases, viseme_ids, weights = data
        n_chars = char_end_idx - char_offset
        if n_chars <= 0:
            return []

        total_w = sum(weights) or 1.0
        out: List[Dict[str, Any]] = []
        cum = 0.0

        for base, vis, w in zip(bases, viseme_ids, weights):
            frac_start = cum / total_w
            cum += w
            frac_end = cum / total_w

            # Fractional interpolation WITHIN characters — keeps every
            # phoneme at a unique, monotonically increasing time even in
            # short words, while still using real per-character timing.
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
                out.append({"t_ms": int(t_start * 1000), "viseme_id": vis})

        return out

    # ----------------------------------------------------------------------
    @staticmethod
    def _char_frac_time(
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
        span = end_times[gidx] - start_times[gidx]
        if span < 0.0:
            span = 0.0
        return start_times[gidx] + frac * span

    # ----------------------------------------------------------------------
    def _letter_fallback(
        self,
        token: str,
        char_offset: int,
        start_times: List[float],
    ) -> List[Dict[str, Any]]:
        """
        Last resort for tokens nothing could phonemize: approximate mouth
        motion from letters, each anchored to that letter's OWN timestamp.
        """
        out: List[Dict[str, Any]] = []
        last_idx = len(start_times) - 1
        for i, ch in enumerate(token):
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
        2. Collision handling (< _MIN_VISEME_GAP_MS apart): closures/stops
           REPLACE the colliding previous shape instead of being dropped —
           a missed M closure is the most visible lipsync error there is.
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
                    prev["viseme_id"] = v["viseme_id"]
                continue

            if v["viseme_id"] == prev["viseme_id"]:
                if v["viseme_id"] in _VISEME_KEEP_REPEAT:
                    smoothed.append(v)
                continue

            smoothed.append(v)

        return smoothed

    # ----------------------------------------------------------------------
    @staticmethod
    def _get_word_windows(characters: List[str]) -> List[Tuple[str, int, int]]:
        """(word, global_char_start_index, global_char_end_index_exclusive)"""
        words: List[Tuple[str, int, int]] = []
        current: List[str] = []
        start_idx: Optional[int] = None

        for i, ch in enumerate(characters):
            if ch in _WHITESPACE:
                if current and start_idx is not None:
                    words.append(("".join(current), start_idx, i))
                current = []
                start_idx = None
            else:
                if start_idx is None:
                    start_idx = i
                current.append(ch)

        if current and start_idx is not None:
            words.append(("".join(current), start_idx, len(characters)))
        return words