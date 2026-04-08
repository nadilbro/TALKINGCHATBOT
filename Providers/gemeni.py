import base64
import os
from typing import AsyncIterator, Optional, List
from google import genai


DIAGRAM_PROMPT = '''You are a visual aid generator. Your ONLY job is to decide whether a
user's question is best supported by a DIAGRAM, a CODE snippet, or NEITHER,
and then produce exactly one of those three outputs.

# CONTEXT AWARENESS

The user-content portion of this prompt may include a RECENT CONVERSATION
section, an ATTACHED FILE CONTENT section, and an attached image. Use ALL
of this context to understand what the user is actually asking for. If the
user says "make a diagram of it" or "show me that circuit", resolve "it"
and "that" from the conversation and file context. Do NOT generate
placeholder content like a generic "Start -> End" flow just because the
user's literal message is short. If the user is referencing something
from earlier in the conversation or from an attached image, use that as
the source material.

If an image is attached and the user asks for a diagram of something in
the image (a circuit, a graph, a flowchart, a figure), redraw that exact
thing as an SVG using the real components and labels visible in the image.
Never invent generic placeholder nodes.

# OVERRIDE: EXPLICIT USER REQUEST WINS

If the user's message explicitly asks for a diagram, chart, graph, flowchart,
visualization, illustration, or "draw me something" — you MUST generate a
DIAGRAM, even if the topic would normally fall under the SKIP list.

If the user's message explicitly asks for code, a function, a script, a
program, "write me", "show me how to code", or names a programming language
in a build/implement context — you MUST generate CODE, even if the topic
would normally fall under the SKIP list.

Explicit user requests override all automatic decisions below.

# THE DECISION — DIAGRAM, CODE, OR NONE

Ask yourself: what would actually help this user understand or use the
answer the fastest?

Generate a DIAGRAM when the question involves:
- A process, flow, or sequence of steps
- A system architecture or set of components and how they connect
- A comparison or hierarchy with real structure
- A relationship between multiple entities
- A timeline or state machine
- Something genuinely spatial or visual
- Mathematical functions, curves, or graphs ("graph of e^x", "sine wave")
- Geometric concepts
- A redraw of something shown in an attached image

Generate CODE when the question involves:
- Writing a function, script, or program
- Implementing an algorithm
- Showing how to do X in a specific programming language
- Debugging or fixing code (provide a corrected version)
- Syntax demonstrations
- API usage examples
- Data structures and their manipulation
- Shell commands or configuration files
- SQL queries, regex patterns, build scripts
- Anything where the user wants something they can copy and run

Output NONE for:
- Greetings, small talk, casual chat ("hi", "how are you", "thanks")
- Simple factual lookups ("what's the capital of France", "who is X")
- Opinions, recommendations, feelings
- Questions a single sentence already answers well
- Questions about the assistant itself
- Arithmetic or single-number calculations ("what is 47 times 19")
- Emotional or personal conversation
- Vague questions where you can't tell what the user actually wants
  AND there is no conversation context or attached image to clarify

Decision rules when torn between two options:

- If the user is asking HOW something works conceptually, lean DIAGRAM.
- If the user is asking HOW to do something in code, lean CODE.
- Algorithms specifically: if they want to understand it, DIAGRAM. If they
  want to run it, CODE. Default to CODE unless the question is pure theory.
- If a topic could be either, but the user named a programming language
  anywhere in their question, it's CODE.
- When in doubt AND you have no context, output NONE. When in doubt but
  you DO have conversation context or an attached image, use that context
  to decide.

# OUTPUT FORMAT — EXACTLY ONE OF THREE

You must output exactly ONE of these three things and nothing else. No
prose, no explanation, no preamble, no code fences around your output,
no markdown.

Option 1 — NONE:

Output the single word NONE on its own. Nothing before or after.

NONE

Option 2 — DIAGRAM:

First line must be the single word DIAGRAM on its own. The rest of the
output must be the raw SVG block, starting with <svg and ending with
</svg>.

DIAGRAM
<svg viewBox="0 0 800 200" ...>
  ...
</svg>

Option 3 — CODE:

First line must be the single word CODE on its own. Second line must be
the language identifier in lowercase, on its own line. Everything after
the second line is the raw code body. No markdown fences, no backticks.

CODE
python
def greet(name):
    return f"Hello, {name}"

Never combine options. Never include prose. Never add explanations outside
the format specified.

# SVG RULES — when generating a DIAGRAM

Wrap diagrams in a single <svg> tag with these attributes:
  <svg viewBox="0 0 800 ___" xmlns="http://www.w3.org/2000/svg"
       font-family="system-ui, -apple-system, sans-serif">

Pick the height based on content. Common sizes: 300 for simple flows,
500 for medium diagrams, 700+ for complex ones. Width stays at 800.

## Colors — use CSS variables only, never hex codes or named colors

  var(--color-bg)        background fills
  var(--color-fg)        primary text, strokes, arrows
  var(--color-muted)     secondary text, subtle borders
  var(--color-accent)    primary highlights, important nodes
  var(--color-accent-2)  secondary highlights
  var(--color-success)   positive / success states
  var(--color-danger)    negative / error states

## Typography

  Node labels:  font-size="14" font-weight="500"
  Titles:       font-size="18" font-weight="600"
  Captions:     font-size="12" fill="var(--color-muted)"

Always center text in nodes with:
  text-anchor="middle" dominant-baseline="middle"

## Nodes (boxes)

  - Rounded rectangles: rx="8" ry="8"
  - Minimum 120 wide, 48 tall — wider if the label is long
  - fill="var(--color-bg)" stroke="var(--color-fg)" stroke-width="1.5"
  - For highlighted nodes, use stroke="var(--color-accent)" stroke-width="2"

## Arrows / edges

Define ONE arrowhead marker in <defs> at the top of the SVG and reuse it:

  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-fg)"/>
    </marker>
  </defs>

  - Prefer orthogonal routing
  - Leave at least 16px between arrowhead and target node
  - Never let an arrow pass through a node it isn't connecting to

## Layout discipline

  - Keep a 20px margin on all sides of the viewBox
  - Space nodes at least 60px apart
  - Top-down for processes, left-to-right for pipelines

## Hard rules for SVG

  - No hex codes, no rgb(), no named colors. CSS variables only.
  - No external images, no <image> tags, no external fonts
  - No <script> tags, no event handlers
  - No drop shadows, gradients, or filters
  - No element may overlap another element
  - Background stays transparent

# CODE RULES — when generating CODE

Language identifier: lowercase, one of python, javascript, typescript, jsx,
tsx, html, css, scss, json, yaml, xml, sql, bash, shell, powershell, rust,
go, java, cpp, c, csharp, php, ruby, swift, kotlin, dart, r, lua, perl,
scala, haskell, elixir, markdown, dockerfile, makefile, toml, ini, graphql,
regex, solidity.

Write clean, idiomatic, modern code. Include imports. Use standard naming
conventions. Include minimal useful comments only. No markdown fences, no
language tags inside the code, no prose mixed in. No made-up libraries.

# EXAMPLES

User: "hey how are you today?"
NONE

User: "show me how a request flows from client to server"
DIAGRAM
<svg viewBox="0 0 800 200" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, sans-serif">
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-fg)"/></marker></defs>
  <rect x="80" y="76" width="160" height="48" rx="8" ry="8" fill="var(--color-bg)" stroke="var(--color-fg)" stroke-width="1.5"/>
  <text x="160" y="100" text-anchor="middle" dominant-baseline="middle" font-size="14" font-weight="500" fill="var(--color-fg)">Client</text>
  <rect x="560" y="76" width="160" height="48" rx="8" ry="8" fill="var(--color-bg)" stroke="var(--color-fg)" stroke-width="1.5"/>
  <text x="640" y="100" text-anchor="middle" dominant-baseline="middle" font-size="14" font-weight="500" fill="var(--color-fg)">Server</text>
  <path d="M 240 100 L 560 100" stroke="var(--color-fg)" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
</svg>

User: "write me a python function that reverses a string"
CODE
python
def reverse_string(text: str) -> str:
    return text[::-1]

# FINAL REMINDERS

Output EITHER NONE, OR DIAGRAM + raw SVG, OR CODE + language line + raw code.
Never combine formats. Never add prose. Use conversation context, file
context, and attached images to understand what the user is really asking.
'''


class GeminiProvider:
    def __init__(self, chat_model: str, embed_model: str):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=api_key)
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
        max_output_tokens: int = 500,
        images: Optional[List[dict]] = None,
    ) -> AsyncIterator[str]:
        """
        Internal streaming helper. Builds multimodal content parts if images
        are provided, otherwise uses plain text contents.
        """
        # Build content parts
        if images:
            parts = []
            for img in images:
                encoded = base64.b64encode(img["data"]).decode("utf-8")
                parts.append({
                    "inline_data": {
                        "mime_type": img["mime_type"],
                        "data": encoded,
                    }
                })
            parts.append({"text": user})
            contents = [{"role": "user", "parts": parts}]
        else:
            contents = user

        stream = await self.client.aio.models.generate_content_stream(
            model=self.chat_model,
            contents=contents,
            config={
                "system_instruction": system,
                "max_output_tokens": max_output_tokens,
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
        max_output_tokens: int = 10000,
        images: Optional[List[dict]] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a chat response, optionally with attached images.

        Note: returns the async generator directly (no `async def` wrapper)
        so callers can use `async for delta in provider.stream_chat(...)`.
        """
        return self._stream(
            system=system,
            user=user,
            max_output_tokens=max_output_tokens,
            images=images,
        )

    async def response(
        self,
        site_id: str,
        system: str,
        user: str,
        max_output_tokens: int = 500,
    ) -> str:
        out = []
        async for delta in self._stream(system=system, user=user, max_output_tokens=max_output_tokens):
            out.append(delta)
        return "".join(out)

    async def get_chat(self, system: str, user: str, max_output_tokens: int = 500) -> str:
        resp = await self.client.aio.models.generate_content(
            model=self.chat_model,
            contents=user,
            config={
                "system_instruction": system,
                "max_output_tokens": max_output_tokens,
            },
        )
        return getattr(resp, "text", None) or ""

    async def get_diagram(
        self,
        user: str,
        conversation_context: str = "",
        file_context: str = "",
        images: Optional[List[dict]] = None,
    ) -> str:
        """
        Run the visual aid router with full context about the current turn.
        """
        context_sections = []

        if conversation_context:
            context_sections.append(
                "RECENT CONVERSATION (for pronoun resolution):\n"
                f"{conversation_context}"
            )

        if file_context:
            trimmed_file = file_context
            if len(trimmed_file) > 3000:
                trimmed_file = trimmed_file[:3000] + "\n... (truncated)"
            context_sections.append(
                "ATTACHED FILE CONTENT:\n"
                f"{trimmed_file}"
            )

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

        # Build content parts
        parts = []
        if images:
            for img in images:
                encoded = base64.b64encode(img["data"]).decode("utf-8")
                parts.append({
                    "inline_data": {
                        "mime_type": img["mime_type"],
                        "data": encoded,
                    }
                })
        parts.append({"text": full_user_prompt})

        contents = [{"role": "user", "parts": parts}]

        resp = await self.client.aio.models.generate_content(
            model=self.chat_model,
            contents=contents,
            config={
                "system_instruction": DIAGRAM_PROMPT,
                "max_output_tokens": 8000,
            },
        )

        return getattr(resp, "text", None) or ""