import os
import base64
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import azure.cognitiveservices.speech as speechsdk


@dataclass
class VisemePoint:
    t_ms: int
    viseme_id: int


class VoiceChatSystem:
    """
    Azure TTS wrapper that returns MP3 bytes + viseme timeline.

    IMPORTANT:
    - Azure's speak_text_async returns a ResultFuture (NOT concurrent.futures.Future).
      So: NO .done() / .result(). Use:
        - future.get()  (blocking)
        - future.wait_for(timeout_ms) then future.get()
    """

    def __init__(self):
        self.speech_key = (os.getenv("AZURE_SPEECH_KEY") or "").strip()
        self.speech_region = (os.getenv("AZURE_SPEECH_REGION") or "").strip()

        if not self.speech_key or not self.speech_region:
            raise RuntimeError(
                "Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION environment variables."
            )

    def _build_speech_config(self, voice_name: str) -> speechsdk.SpeechConfig:
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region,
        )
        if voice_name:
            speech_config.speech_synthesis_voice_name = voice_name

        # MP3 output (works well with browser MSE)
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        return speech_config

    def synthesize_mp3_with_visemes(
        self,
        text: str,
        voice_name: str,
        timeout_ms: int = 20000,
    ) -> Tuple[bytes, List[VisemePoint]]:
        """
        Blocking synthesis. Use run_in_threadpool() in FastAPI for async endpoints.
        Returns (mp3_bytes, viseme_points).
        """
        text = (text or "").strip()
        if not text:
            return b"", []

        speech_config = self._build_speech_config(voice_name)

        # Use a "null" audio output — we only want bytes from result.audio_data.
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        visemes: List[VisemePoint] = []

        # Viseme timestamps come as "audio_offset" in 100-ns units.
        def on_viseme(evt: speechsdk.SpeechSynthesisVisemeEventArgs):
            try:
                t_ms = int(evt.audio_offset / 10_000)  # 100ns -> ms
                visemes.append(VisemePoint(t_ms=t_ms, viseme_id=int(evt.viseme_id)))
            except Exception:
                # Don’t let event parsing kill synthesis
                pass

        synthesizer.viseme_received.connect(on_viseme)

        future = synthesizer.speak_text_async(text)

        # IMPORTANT: ResultFuture has no .done()
        if not future.wait_for(timeout_ms):
            # cancel + raise
            try:
                synthesizer.stop_speaking_async()
            except Exception:
                pass
            raise TimeoutError(f"TTS timed out after {timeout_ms}ms")

        result: speechsdk.SpeechSynthesisResult = future.get()

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = ""
            try:
                details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result).error_details
            except Exception:
                pass
            raise RuntimeError(f"TTS failed: {result.reason}. {details}".strip())

        audio_bytes = bytes(result.audio_data) if result.audio_data else b""
        visemes.sort(key=lambda v: v.t_ms)

        return audio_bytes, visemes

    @staticmethod
    def bytes_to_b64(data: bytes) -> str:
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def chunk_bytes(data: bytes, chunk_size: int = 8192):
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]
