import os
import json
import base64
from typing import List, Dict, Any, Tuple

import azure.cognitiveservices.speech as speechsdk
from fastapi.concurrency import run_in_threadpool


class VoiceChatSystem:
    def __init__(self):
        self.speech_key = (os.getenv("AZURE_SPEECH_KEY") or "").strip()
        self.speech_region = (os.getenv("AZURE_SPEECH_REGION") or "").strip()

        if not self.speech_key or not self.speech_region:
            raise RuntimeError("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION env vars")

    async def synthesize_mp3_with_visemes(
        self,
        text: str,
        voice_name: str,
    ) -> Tuple[bytes, List[Dict[str, Any]]]:
        """
        Returns:
          - mp3_bytes
          - visemes: [{ "t_ms": int, "viseme_id": int }]
        """

        def _blocking():
            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                region=self.speech_region
            )

            # MP3 output for browser streaming
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
            )

            if voice_name:
                speech_config.speech_synthesis_voice_name = voice_name

            # Collect visemes
            visemes: List[Dict[str, Any]] = []

            # IMPORTANT: no AudioConfig => SDK returns audio_data bytes in result
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

            def on_viseme(evt: speechsdk.SpeechSynthesisVisemeEventArgs):
                # audio_offset is in 100-nanosecond units
                t_ms = int(evt.audio_offset / 10_000)
                visemes.append({"t_ms": t_ms, "viseme_id": int(evt.viseme_id)})

            synthesizer.viseme_received.connect(on_viseme)

            # Azure returns ResultFuture — use .get(), NOT .done()
            future = synthesizer.speak_text_async(text)
            result = future.get()

            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Try pull error details
                details = ""
                try:
                    details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result).error_details
                except Exception:
                    pass
                raise RuntimeError(f"TTS failed: {result.reason} {details}".strip())

            audio_bytes = result.audio_data or b""
            return audio_bytes, visemes

        return await run_in_threadpool(_blocking)
