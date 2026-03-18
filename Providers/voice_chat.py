import os
from typing import List, Dict, Any, Tuple
from elevenlabs.client import AsyncElevenLabs
from Providers.phenome_provider import TextVisemeProvider
from elevenlabs.client import ElevenLabs
from fastapi.concurrency import run_in_threadpool

class VoiceChatSystem:
    def __init__(self):
        self.api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
        self.voice_id = (os.getenv("ELEVENLABS_VOICE_ID") or "").strip()

        if not self.api_key:
            raise RuntimeError("Missing ELEVENLABS_API_KEY env var")
        if not self.voice_id:
            raise RuntimeError("Missing ELEVENLABS_VOICE_ID env var")

        self.client = ElevenLabs(api_key=self.api_key)
        self.viseme_provider = TextVisemeProvider()

    async def synthesize_mp3_with_visemes(
        self,
        text: str,
        voice_name: str = None,  # kept for API compatibility, unused
    ) -> Tuple[bytes, List[Dict[str, Any]]]:
        """
        Returns:
          - mp3_bytes  (from ElevenLabs)
          - visemes:   [{ "t_ms": int, "viseme_id": int }]  (from TextVisemeProvider)
        """

        # Get audio from ElevenLabs
        audio_bytes = await run_in_threadpool(self._get_elevenlabs_audio, text)
        visemes = self.viseme_provider.get_visemes(text)
        return audio_bytes, visemes

    def _get_elevenlabs_audio(self, text: str) -> bytes:
        response = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        chunks = []
        for chunk in response:
            if isinstance(chunk, bytes):
                chunks.append(chunk)
        return b"".join(chunks)