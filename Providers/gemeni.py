from typing import AsyncIterator, Optional
from google import genai
import os

class GeminiProvider:
    def __init__(self, chat_model: str, embed_model: str):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        

        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.chat_model = chat_model
        self.embed_model = embed_model

    async def embed(self, text: str) -> list[float]:
        resp = await self.client.aio.models.embed_content(
            model=self.embed_model,
            contents=text,
        )
        emb0 = resp.embeddings[0]
        values = getattr(emb0, "values", None) or emb0["values"]
        return list(values)

    async def _stream(
        self,
        system: str,
        user: str,
        max_output_tokens: int = 500
    ) -> AsyncIterator[str]:

        stream = await self.client.aio.models.generate_content_stream(
            model=self.chat_model,
            contents=user,
            config={
                "system_instruction": system,
            },
        )

        async for chunk in stream:
            txt = getattr(chunk, "text", None)
            if txt:
                yield txt


    def stream_chat(
        self,
        system: str,
        user: str,
        max_output_tokens: int = 120
    ) -> AsyncIterator[str]:
        return self._stream(system=system, user=user, max_output_tokens=max_output_tokens)
