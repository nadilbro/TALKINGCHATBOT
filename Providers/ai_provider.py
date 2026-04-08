import base64
from typing import AsyncIterator
from Providers.open_ai import OpenAIProvider
from Providers.gemeni import GeminiProvider
from SQL.SQLManager import VectorRAGService


class AIProvider:
    def __init__(self, rag: VectorRAGService):
        self.rag = rag

        self._providers = {
            "openai": OpenAIProvider(
                chat_model="gpt-5-nano",
                embed_model="text-embedding-3-small",
            ),
            "gemini_flash": GeminiProvider(
                chat_model="gemini-2.5-flash",
                embed_model="gemini-embedding-001",
            ),
            "gemini_diagram": GeminiProvider(
                chat_model="gemini-2.5-flash", # input $2 per million tokens output #gemini_pro gemini-2.5-pro
                embed_model="gemini-embedding-001", 
            ),
            "gemini_pro": GeminiProvider(
                chat_model="gemini-2.5-pro", #gemini_pro gemini-2.5-pro
                embed_model="gemini-embedding-001",
            ),
            "gemini_image": GeminiProvider(
                chat_model="gemini-2.5-flash",  # vision-capable
                embed_model="gemini-embedding-001",
            ),
        }

    async def _tenant_chat_provider_name(self, site_id: str) -> str:
        pro_bool = self.rag.get_pro_usage(site_id)
        if pro_bool:
            return "gemini_pro"
        else: 
            return "gemini_flash"
        
    async def _tenant_diagram_provider_name(self, site_id: str) -> str:
        pro_bool = self.rag.get_pro_usage(site_id)
        if pro_bool:
            return "gemini_pro"
        else: 
            return "gemini_diagram"

    async def stream(self, site_id: str, system: str, user: str) -> AsyncIterator[str]:
        provider_name = await self._tenant_chat_provider_name(site_id)
        provider = self._providers[provider_name]
        async for delta in provider.stream_chat(system=system, user=user, max_output_tokens=10000):
            yield delta

    async def chat(self, site_id: str, system: str, user: str) -> str:
        provider_name = await self._tenant_chat_provider_name(site_id)
        provider = self._providers[provider_name]
        return await provider.response(site_id=site_id, system=system, user=user)

    async def get_diagram(self, site_id: str, user: str) -> str:
        provider_name = await self._tenant_diagram_provider_name(site_id)
        provider = self._providers[provider_name]
        return await provider.get_diagram(user=user)

    async def extract_image_text(self, file_bytes: bytes, ext: str) -> str:
        """
        Calls Gemini Vision at upload time to extract text/description from an image.
        Returns plain text — goes straight into your RAG chunking pipeline.
        """
        mime_map = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
        }
        mime_type = mime_map.get(ext.lower())
        if not mime_type:
            raise ValueError(f"Unsupported image type: {ext}")

        provider = self._providers["gemini_image"]

        prompt = (
            "Extract all text and information from this image in full detail. "
            "If it contains diagrams, charts, or visual data, describe them clearly. "
            "Format the output as clean readable text."
        )

        encoded = base64.b64encode(file_bytes).decode("utf-8")

        resp = await provider.client.aio.models.generate_content(
            model=provider.chat_model,
            contents=[
                {
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": encoded}},
                        {"text": prompt},
                    ]
                }
            ],
        )
        return getattr(resp, "text", None) or ""