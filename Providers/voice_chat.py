import os
import asyncio
from typing import List, Dict, Any, Tuple
import azure.cognitiveservices.speech as speechsdk
from rhubarb_provider import RhubarbProvider
from fastapi.concurrency import run_in_threadpool


class VoiceChatSystem:
    def __init__(self):
        self.speech_key = (os.getenv("AZURE_SPEECH_KEY") or "").strip()
        self.speech_region = (os.getenv("AZURE_SPEECH_REGION") or "").strip()

        if not self.speech_key or not self.speech_region:
            raise RuntimeError("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION env vars")
        
        self.rhubarb = RhubarbProvider() #Rubharb

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

        audio_task = asyncio.create_task(
            run_in_threadpool(self._azure_synthesize, text, voice_name)
        )
        viseme_task = asyncio.create_task(
            self.rhubarb.from_text_only(text)
        )
 
        # Wait for both to finish
        audio_bytes, visemes = await asyncio.gather(audio_task, viseme_task)
 
        return audio_bytes, visemes
    
    def _azure_synthesize(self, text: str, voice_name: str) -> bytes:
        """Blocking Azure TTS call — runs in threadpool."""
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region,
        )
 
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
 
        if voice_name:
            speech_config.speech_synthesis_voice_name = voice_name
 
        # No AudioConfig => audio comes back as bytes in result
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None,
        )
 
        result = synthesizer.speak_text_async(text).get()
 
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = ""
            try:
                details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result).error_details
            except Exception:
                pass
            raise RuntimeError(f"TTS failed: {result.reason} {details}".strip())
 
        return result.audio_data or b""