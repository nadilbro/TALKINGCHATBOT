from typing import AsyncIterator
from Providers.open_ai import OpenAIProvider
from Providers.gemeni import GeminiProvider
from SQL.RAG import VectorRAGService

class AIProvider:
    def __init__(self, rag: VectorRAGService):
        self.rag = rag

        # cache provider instances (don’t recreate clients per request)
        self._providers = {
            "openai": OpenAIProvider(chat_model="gpt-5-nano", embed_model="text-embedding-3-small"),
            "gemini": GeminiProvider(chat_model="gemini-2.0-flash", embed_model="gemini-2.0-pro"),
        }


    async def _tenant_provider_name(self, site_id: str) -> str: #TEMPERORY
        return "gemini" 

    async def stream(self, site_id: str, system: str, user: str) -> AsyncIterator[str]:
        provider_name = await self._tenant_provider_name(site_id)
        provider = self._providers[provider_name]

        async for delta in provider.stream_chat(system=system, user=user, max_output_tokens=10000):
            yield delta

    async def chat(self, site_id: str, system: str, user: str) -> str:
        provider_name = await self._tenant_provider_name(site_id)
        provider = self._providers[provider_name]
        
        return await provider.response(site_id=site_id, system=system, user=user)

        