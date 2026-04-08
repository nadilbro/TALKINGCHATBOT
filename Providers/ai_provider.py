import base64
from typing import AsyncIterator, Optional, List
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
            # Default chat model — fast, cheap, multimodal
            "gemini_flash": GeminiProvider(
                chat_model="gemini-2.5-flash",
                embed_model="gemini-embedding-001",
            ),
            # Diagram / code generation — always Flash regardless of pro mode.
            "gemini_diagram": GeminiProvider(
                chat_model="gemini-2.5-flash",
                embed_model="gemini-embedding-001",
            ),
            # Pro mode — stronger reasoning for the CHAT response only.
            "gemini_pro": GeminiProvider(
                chat_model="gemini-2.5-pro",
                embed_model="gemini-embedding-001",
            ),
            # Image understanding (vision input) — multimodal Flash
            "gemini_image": GeminiProvider(
                chat_model="gemini-2.5-flash",
                embed_model="gemini-embedding-001",
            ),
        }

    async def _tenant_chat_provider_name(self, site_id: str) -> str:
        """Chat response model — honors pro mode for higher quality answers."""
        pro_bool = self.rag.get_pro_usage(site_id)
        if pro_bool:
            return "gemini_pro"
        return "gemini_flash"

    async def _tenant_diagram_provider_name(self, site_id: str) -> str:
        """Diagram model — ALWAYS Flash, regardless of pro mode."""
        return "gemini_diagram"

    async def stream(
        self,
        site_id: str,
        system: str,
        user: str,
        images: Optional[List[dict]] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a chat response from the correct tenant provider, optionally
        with attached images for vision input.

        Args:
            site_id: User ID, used to pick flash vs pro
            system: System prompt
            user: User message + conversation history
            images: Optional list of {"mime_type": str, "data": bytes}
                    where `data` is raw image bytes (not base64).
        """
        provider_name = await self._tenant_chat_provider_name(site_id)
        provider = self._providers[provider_name]
        async for delta in provider.stream_chat(
            system=system,
            user=user,
            max_output_tokens=10000,
            images=images,
        ):
            yield delta

    async def chat(self, site_id: str, system: str, user: str) -> str:
        provider_name = await self._tenant_chat_provider_name(site_id)
        provider = self._providers[provider_name]
        return await provider.response(site_id=site_id, system=system, user=user)

    async def get_diagram(
        self,
        site_id: str,
        user: str,
        conversation_context: str = "",
        file_context: str = "",
        images: Optional[List[dict]] = None,
    ) -> str:
        """
        Run the visual aid router with full context about the current turn.

        Args:
            site_id: User ID for provider selection
            user: The user's latest message
            conversation_context: Last few turns so pronouns resolve
            file_context: Attached text file content (PDF, DOCX, etc.)
            images: Optional list of attached images for the router to see
        """
        provider_name = await self._tenant_diagram_provider_name(site_id)
        provider = self._providers[provider_name]
        return await provider.get_diagram(
            user=user,
            conversation_context=conversation_context,
            file_context=file_context,
            images=images,
        )

    async def extract_image_text(self, file_bytes: bytes, ext: str) -> str:
        """
        DEPRECATED — kept for backwards compatibility.

        Previously used to convert an uploaded image to text via a separate
        Gemini call. The new approach passes image bytes directly to the
        chat call as multimodal input. Use FileExtractor.is_image() to
        detect images, then build an images list and pass it to ai.stream().
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
            "Transcribe this image exactly as it appears. Preserve all text verbatim "
            "and describe any figures, circuits, or graphs in enough technical detail "
            "that the problem could be solved from your description alone."
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