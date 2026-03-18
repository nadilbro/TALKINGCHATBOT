import os
from typing import List, Dict, Any, Tuple
from elevenlabs.client import AsyncElevenLabs
from Providers.phenome_provider import TextVisemeProvider


class VoiceChatSystem:
    def __init__(self):
        self.api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
        self.voice_id = (os.getenv("ELEVENLABS_VOICE_ID") or "").strip()

        if not self.api_key:
            raise RuntimeError("Missing ELEVENLABS_API_KEY env var")
        if not self.voice_id:
            raise RuntimeError("Missing ELEVENLABS_VOICE_ID env var")

        self.client = AsyncElevenLabs(api_key=self.api_key)
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
        audio_bytes = await self._get_elevenlabs_audio(text)

        # Generate visemes instantly from text — no audio processing needed
        visemes = self.viseme_provider.get_visemes(text)

        return audio_bytes, visemes

    async def _get_elevenlabs_audio(self, text: str) -> bytes:
        audio_generator = await self.client.generate(
            text=text,
            voice=self.voice_id,
            model="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        chunks = []
        async for chunk in audio_generator:
            if chunk:
                chunks.append(chunk)
        return b"".join(chunks)