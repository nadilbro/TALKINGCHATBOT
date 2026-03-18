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
    "AA": 3, "AE": 3, "AH": 3, "AO": 3,
    "AW": 4, "AY": 3, "EH": 2, "ER": 2,
    "EY": 2, "IH": 2, "IY": 2, "OW": 4,
    "OY": 4, "UH": 4, "UW": 4,
    # Consonants
    "B":  1, "CH": 7, "D":  8, "DH": 6,
    "F":  5, "G":  8, "HH": 8, "JH": 7,
    "K":  8, "L":  8, "M":  1, "N":  8,
    "NG": 8, "P":  1, "R":  8, "S":  8,
    "SH": 7, "T":  8, "TH": 6, "V":  5,
    "W":  4, "Y":  2, "Z":  8, "ZH": 7,
}

# Average phoneme duration in milliseconds (rough estimate for natural speech)
PHONEME_DURATION_MS = 80


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
        """
        Convert text to a list of timed viseme events.

        Args:
            text: The transcript string (e.g. Gemini's response)

        Returns:
            List of { "t_ms": int, "viseme_id": int }
        """
        words = self._tokenize(text)
        visemes = []
        t_ms = 0

        for word in words:
            phonemes = self._word_to_phonemes(word)

            if not phonemes:
                # Unknown word — add a short pause and move on
                t_ms += PHONEME_DURATION_MS * 2
                continue

            for phoneme in phonemes:
                viseme_id = ARPABET_TO_VISEME.get(phoneme, 8)
                visemes.append({"t_ms": t_ms, "viseme_id": viseme_id})
                t_ms += PHONEME_DURATION_MS

            # Short pause between words
            t_ms += PHONEME_DURATION_MS

        # Always end on silence
        visemes.append({"t_ms": t_ms, "viseme_id": 0})

        return visemes

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