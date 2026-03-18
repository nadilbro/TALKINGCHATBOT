import os
import json
import tempfile
import subprocess
from typing import List, Dict, Any
from fastapi.concurrency import run_in_threadpool
import sys

# Preston Blair viseme label → ID
# A = silence/rest,  B = p/b/m,  C = ee/ih,  D = oh
# E = oo,            F = f/v,    G = th/dh,  H = ch/sh/j  X = s/z/t/d/k/g
RHUBARB_LABEL_TO_ID = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
    "F": 5, "G": 6, "H": 7, "X": 8
}


class RhubarbProvider:
    """
    Lightweight wrapper around the Rhubarb Lip Sync CLI binary.
    Extracts viseme timestamps from either:
      - audio file + transcript  (most accurate)
      - transcript only          (fast/rough, good enough for casual lip sync)
    """

    def __init__(self, timeout: int = 30):
        # Points to the rhubarb binary in the project root
        self.rhubarb_bin = "rhubarb"
        self.timeout = timeout


    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def from_audio_and_text(
        self,
        audio_bytes: bytes,
        text: str,
        audio_suffix: str = ".mp3",
    ) -> List[Dict[str, Any]]:
        """
        Most accurate mode — give it the real audio + transcript.
        Rhubarb aligns the phonemes to the actual audio timing.

        Returns: [{ "t_ms": int, "viseme_id": int }, ...]
        """
        return await run_in_threadpool(
            self._run, audio_bytes, text, audio_suffix
        )

    async def from_text_only(self, text: str) -> List[Dict[str, Any]]:
        """
        Fast/rough mode — no audio needed.
        Rhubarb uses espeak internally to estimate timing.
        Good enough for 'looks alive' lip sync.

        Returns: [{ "t_ms": int, "viseme_id": int }, ...]
        """
        return await run_in_threadpool(self._run, None, text, None)

    # ------------------------------------------------------------------
    # Internal blocking implementation (runs in threadpool)
    # ------------------------------------------------------------------
    
    #AI code to get rubharb working
    def _run(
        self,
        audio_bytes: bytes | None,
        text: str,
        audio_suffix: str | None,
    ) -> List[Dict[str, Any]]:

        tmp_files = []

        try:
            # Write transcript to a temp file (always needed)
            text_f = tempfile.NamedTemporaryFile(
                suffix=".txt", delete=False, mode="w", encoding="utf-8"
            )
            text_f.write(text)
            text_f.close()
            tmp_files.append(text_f.name)

            cmd = [self.rhubarb_bin, "-f", "json", "--recognizer", "phonetic"]

            if audio_bytes:
                audio_f = tempfile.NamedTemporaryFile(
                    suffix=audio_suffix, delete=False
                )
                audio_f.write(audio_bytes)
                audio_f.close()
                tmp_files.append(audio_f.name)
                cmd += ["--dialogFile", text_f.name, audio_f.name]
            else:
                cmd.append(text_f.name)

            # Debug AFTER cmd is built
            print(f"==> CMD: {' '.join(cmd)}", file=sys.stderr, flush=True)
            espeak_check = subprocess.run(["espeak", "--version"], capture_output=True, text=True)
            print(f"==> espeak: {espeak_check.stdout} {espeak_check.stderr}", file=sys.stderr, flush=True)
            espeak_ng_check = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True)
            print(f"==> espeak-ng: {espeak_ng_check.stdout} {espeak_ng_check.stderr}", file=sys.stderr, flush=True)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={
                    **os.environ,
                    "ESPEAK_DATA_PATH": "/usr/lib/x86_64-linux-gnu/espeak-data",
                }
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Rhubarb exited with code {result.returncode}: {result.stderr.strip()}"
                )

            data = json.loads(result.stdout)
            return [
                {
                    "t_ms": int(float(cue["start"]) * 1000),
                    "viseme_id": RHUBARB_LABEL_TO_ID.get(cue["value"], 0),
                }
                for cue in data.get("mouthCues", [])
            ]

        finally:
            for path in tmp_files:
                try:
                    os.unlink(path)
                except OSError:
                    pass