from typing import AsyncIterator
from Providers.open_ai import OpenAIProvider
from Providers.gemeni import GeminiProvider
from SQL.SQLManager import VectorRAGService


class AIProvider:
    def __init__(self, rag: VectorRAGService):
        self.rag = rag

        # Two Gemini instances pointing at different models.
        # Flash for fast spoken responses, Pro for high-quality diagrams.
        self._providers = {
            "openai": OpenAIProvider(
                chat_model="gpt-5-nano",
                embed_model="text-embedding-3-small",
            ),
            "gemini_flash": GeminiProvider(
                chat_model="gemini-2.5-flash",          # ← verify with Google's docs
                embed_model="gemini-embedding-001",     # ← verify with Google's docs
            ),
            "gemini_pro": GeminiProvider(
                chat_model="gemini-3-flash-preview",            # ← verify with Google's docs
                embed_model="gemini-embedding-001",     # ← verify with Google's docs
            ),
        }

    async def _tenant_chat_provider_name(self, site_id: str) -> str:
        # TODO: per-user model selection
        return "gemini_flash"

    async def _tenant_diagram_provider_name(self, site_id: str) -> str:
        # TODO: per-user model selection
        return "gemini_pro"

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