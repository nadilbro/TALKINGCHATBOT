import re
from typing import List, Dict, Any
import nltk
from nltk.corpus import cmudict

# Download CMU dict if not already present
nltk.download("cmudict", quiet=True)

# ---------------------------------------------------------------------------
# ARPAbet phoneme → Preston Blair viseme ID (0–8)
# ---------------------------------------------------------------------------
# 0 = A  silence/rest
# 1 = B  p, b, m
# 2 = C  ee, ih
# 3 = D  oh, ah, aa
# 4 = E  oo, ow, uw
# 5 = F  f, v
# 6 = G  th, dh
# 7 = H  ch, sh, j, zh
# 8 = X  s, z, t, d, k, g, n, l, r (default consonants)
# ---------------------------------------------------------------------------
ARPABET_TO_VISEME = {
    # Vowels
    "AA": 2,  "AE": 1,  "AH": 1,  "AO": 3,
    "AW": 9,  "AY": 11, "EH": 4,  "ER": 5,
    "EY": 11, "IH": 6,  "IY": 6,  "OW": 8,
    "OY": 10, "UH": 4,  "UW": 7,
    # Consonants
    "B":  21, "CH": 16, "D":  19, "DH": 17,
    "F":  18, "G":  20, "HH": 12, "JH": 16,
    "K":  20, "L":  14, "M":  21, "N":  19,
    "NG": 20, "P":  21, "R":  13, "S":  15,
    "SH": 16, "T":  19, "TH": 17, "V":  18,
    "W":  7,  "Y":  6,  "Z":  15, "ZH": 16,
}

# Average phoneme duration in milliseconds (rough estimate for natural speech)
PHONEME_DURATION_MS = 25


class TextVisemeProvider:
    """
    Generates viseme timestamps directly from text using NLTK's CMU
    Pronouncing Dictionary. No external binaries, no audio needed.
    Runs in under 50ms — suitable for real-time conversational AI.

    Output is compatible with the existing websocket viseme format:
        [{ "t_ms": int, "viseme_id": int }, ...]
    """

    def __init__(self):
        self._cmu = cmudict.dict()

    def get_visemes(self, text: str) -> List[Dict[str, Any]]:
        words = self._tokenize(text)
        visemes = []
        t_ms = 0

        for word in words:
            phonemes = self._word_to_phonemes(word)
            if not phonemes:
                t_ms += PHONEME_DURATION_MS * 2
                continue
            for phoneme in phonemes:
                viseme_id = ARPABET_TO_VISEME.get(phoneme, 19)
                visemes.append({"t_ms": t_ms, "viseme_id": viseme_id})
                t_ms += PHONEME_DURATION_MS
            t_ms += PHONEME_DURATION_MS

        visemes.append({"t_ms": t_ms, "viseme_id": 0})

        # Scale to estimated audio duration based on word count
        # ElevenLabs flash speaks at roughly 150 words per minute
        word_count = len(words)
        estimated_duration_ms = (word_count / 150) * 60 * 1000
        
        if t_ms > 0 and estimated_duration_ms > 0:
            scale = estimated_duration_ms / t_ms
            visemes = [
                {"t_ms": int(v["t_ms"] * scale), "viseme_id": v["viseme_id"]}
                for v in visemes
            ]

        # Remove consecutive duplicates
        deduped = []
        for v in visemes:
            if not deduped or v["viseme_id"] != deduped[-1]["viseme_id"]:
                deduped.append(v)

        return deduped
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase and split text into words, stripping punctuation."""
        text = text.lower()
        text = re.sub(r"[^a-z\s']", "", text)
        return [w for w in text.split() if w]

    def _word_to_phonemes(self, word: str) -> List[str]:
        """
        Look up a word in the CMU dict and return its phonemes.
        Strips stress digits (e.g. 'AH0' -> 'AH').
        Falls back to None if the word isn't in the dictionary.
        """
        entries = self._cmu.get(word)
        if not entries:
            return []

        # Take the first pronunciation
        return [re.sub(r"\d", "", p) for p in entries[0]]