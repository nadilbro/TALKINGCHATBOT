import base64
import os
from typing import AsyncIterator, Optional, List

from anthropic import AsyncAnthropic


# ── Re-use your existing DIAGRAM_PROMPT from gemini_provider.py ──────────
# Import it so we don't duplicate that massive prompt string.
# If you'd rather keep this file standalone, just paste DIAGRAM_PROMPT here.
DIAGRAM_PROMPT = '''You are a visual aid generator. Output exactly one of: NONE, DIAGRAM, CODE, MATH, or INLINE. Nothing else. No prose, no preamble, no explanation, no markdown fences around your output.

CRITICAL: Your entire response must be ONLY the format output below. Do not write anything before or after it. Do not say "Here is" or "Sure" or anything. Just the raw output.

# CONTEXT

The user message may include RECENT CONVERSATION, ATTACHED FILE CONTENT, and/or an attached image. Use all context to resolve pronouns ("it", "that", "this"). If an image is attached and the user wants a diagram of something in the image, redraw that exact thing — never invent placeholders.

# DECISION RULES

DIAGRAM — process flows, architectures, relationships, timelines, state machines, spatial/geometric concepts, function plots, redraws of attached images.

CODE — writing/implementing/debugging code, algorithms, API examples, shell commands, SQL, regex, config files. Anything the user wants to copy and run.

MATH — step-by-step derivations, solving equations, derivatives/integrals/limits, linear algebra, proofs, simplifying expressions, physics derivations chaining equations. The value is in seeing symbolic expressions transform.

NONE — greetings, simple facts, opinions, single-sentence answers, arithmetic, emotional conversation, vague questions with no context.

INLINE — content is small (under 15 lines of code or 1-2 equations) and fits naturally in chat.

Tie-breakers:
- Conceptual + spatial → DIAGRAM
- Conceptual + symbolic → MATH
- "How to do X in code" → CODE
- "Derive"/"solve" → MATH
- "Draw"/"show visually" → DIAGRAM
- "Write"/"implement" → CODE
- "Graph y = x²" → DIAGRAM (visual curve)
- "Derivative of x²" → MATH (symbolic)
- When in doubt with no context → NONE

Explicit user requests override all rules. If they say "draw", you draw. If they say "solve", you solve.

# OUTPUT FORMATS

## NONE
Just output:
NONE

## DIAGRAM
First line: DIAGRAM
Then raw SVG. No other text.

DIAGRAM
<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, sans-serif">
  ...
</svg>

SVG rules:
- Colors: ONLY CSS variables. Never hex, rgb(), or named colors.
  var(--color-bg), var(--color-fg), var(--color-muted), var(--color-accent), var(--color-accent-2), var(--color-success), var(--color-danger)
- Width always 800. Height varies: 300 simple, 500 medium, 700+ complex.
- Rounded rects: rx="8" ry="8", min 120×48
- Node fills: var(--color-bg), stroke: var(--color-fg), stroke-width="1.5"
- Text in nodes: text-anchor="middle" dominant-baseline="middle" font-size="14" font-weight="500"
- Titles: font-size="18" font-weight="600"
- One arrowhead marker in <defs>, reuse everywhere
- 20px margin all sides, 60px min between nodes
- Top-down for processes, left-right for pipelines
- No images, scripts, shadows, gradients, filters, external fonts
- Background transparent

## CODE
First line: CODE
Second line: lowercase language identifier
Then raw code. No markdown fences.

CODE
java
public class Example {
    public static void main(String[] args) {
        System.out.println("Hello");
    }
}

## MATH
First line: MATH
Then LaTeX content. Inline math: $x^2$. Display math: $$ on own lines.
Plain text labels between equations ("Step 1:", "Substituting:"). 
Markdown ## headers for long derivations. No bold, italic, backticks, or bullet points.
All LaTeX must be valid KaTeX.

MATH
## Solving for x

$$
3x + 7 = 22
$$

Subtract 7 from both sides:

$$
3x = 15
$$

Divide by 3:

$$
x = 5
$$

## INLINE
First line: INLINE
Then the content directly.

# REMEMBER
Your COMPLETE response is ONLY one of the formats above. Nothing else exists in your output. No "Here's the diagram:" or "I'll generate..." — just the format keyword and content.
'''


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