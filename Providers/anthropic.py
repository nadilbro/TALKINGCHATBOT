import base64
import os
from typing import AsyncIterator, Optional, List

from anthropic import AsyncAnthropic


# ── Re-use your existing DIAGRAM_PROMPT from gemini_provider.py ──────────
# Import it so we don't duplicate that massive prompt string.
# If you'd rather keep this file standalone, just paste DIAGRAM_PROMPT here.
from gemeni import DIAGRAM_PROMPT


class AnthropicProvider:
    """
    Drop-in replacement for GeminiProvider using Claude Sonnet.
    Same method signatures: stream_chat, response, get_chat, get_diagram, embed.
    """

    def __init__(self, chat_model: str = "claude-sonnet-4-6", embed_model: str = ""):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        self.client = AsyncAnthropic(api_key=api_key)
        self.chat_model = chat_model
        self.embed_model = embed_model  # Anthropic doesn't have embeddings — see embed()

    # ──────────────────────────────────────────────────────────────────────
    # Embeddings — Anthropic doesn't offer an embedding model.
    # Stub this so the interface matches. You can keep using Gemini's
    # embed model, or swap in OpenAI/Voyage embeddings here.
    # ──────────────────────────────────────────────────────────────────────
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Anthropic does not provide an embedding model. "
            "Keep using your Gemini embed_model or swap in Voyage/OpenAI."
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal: build the messages list with optional images
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_user_content(
        user: str,
        images: Optional[List[dict]] = None,
    ) -> list | str:
        """
        Build the `content` value for a user message.
        If images are present, returns a list of content blocks.
        Otherwise returns the plain string (Anthropic accepts both).
        """
        if not images:
            return user

        blocks: list[dict] = []
        for img in images:
            encoded = base64.b64encode(img["data"]).decode("utf-8")
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img["mime_type"],
                    "data": encoded,
                },
            })
        blocks.append({"type": "text", "text": user})
        return blocks

    # ──────────────────────────────────────────────────────────────────────
    # Streaming (async generator — same shape as GeminiProvider._stream)
    # ──────────────────────────────────────────────────────────────────────
    async def _stream(
        self,
        system: str,
        user: str,
        max_output_tokens: int = 500,
        images: Optional[List[dict]] = None,
    ) -> AsyncIterator[str]:
        content = self._build_user_content(user, images)

        async with self.client.messages.stream(
            model=self.chat_model,
            max_tokens=max_output_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    def stream_chat(
        self,
        system: str,
        user: str,
        max_output_tokens: int = 10000,
        images: Optional[List[dict]] = None,
    ) -> AsyncIterator[str]:
        """
        Matches GeminiProvider.stream_chat — returns the async generator
        directly so callers do `async for delta in provider.stream_chat(...)`.
        """
        return self._stream(
            system=system,
            user=user,
            max_output_tokens=max_output_tokens,
            images=images,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Non-streaming response (collects full text)
    # ──────────────────────────────────────────────────────────────────────
    async def response(
        self,
        site_id: str,
        system: str,
        user: str,
        max_output_tokens: int = 500,
    ) -> str:
        out: list[str] = []
        async for delta in self._stream(
            system=system, user=user, max_output_tokens=max_output_tokens
        ):
            out.append(delta)
        return "".join(out)

    async def get_chat(
        self,
        system: str,
        user: str,
        max_output_tokens: int = 500,
    ) -> str:
        """
        Non-streaming single-shot. Used for summaries, file extraction, etc.
        """
        msg = await self.client.messages.create(
            model=self.chat_model,
            max_tokens=max_output_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # msg.content is a list of content blocks
        return "".join(
            block.text for block in msg.content if hasattr(block, "text")
        )

    # ──────────────────────────────────────────────────────────────────────
    # Diagram / visual aid router (matches GeminiProvider.get_diagram)
    # ──────────────────────────────────────────────────────────────────────
    async def get_diagram(
        self,
        user: str,
        conversation_context: str = "",
        file_context: str = "",
        images: Optional[List[dict]] = None,
    ) -> str:
        context_sections: list[str] = []

        if conversation_context:
            context_sections.append(
                f"RECENT CONVERSATION (for pronoun resolution):\n{conversation_context}"
            )

        if file_context:
            trimmed = file_context[:3000] + "\n... (truncated)" if len(file_context) > 3000 else file_context
            context_sections.append(f"ATTACHED FILE CONTENT:\n{trimmed}")

        if images:
            context_sections.append(
                "An image is attached to this turn. If the user is asking "
                "for a diagram of something shown in the image (a circuit, "
                "graph, or figure), redraw it as an SVG using the real "
                "components and labels visible in the image. Do not invent "
                "generic placeholder content."
            )

        context_sections.append(f"LATEST USER MESSAGE:\n{user}")
        full_user_prompt = "\n\n".join(context_sections)

        # Build content blocks (images + text)
        content = self._build_user_content(full_user_prompt, images)

        msg = await self.client.messages.create(
            model=self.chat_model,
            max_tokens=8000,
            system=DIAGRAM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )

        return "".join(
            block.text for block in msg.content if hasattr(block, "text")
        )