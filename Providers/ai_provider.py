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
            # Default chat model — fast, cheap, multimodal
            "gemini_flash": GeminiProvider(
                chat_model="gemini-2.5-flash",
                embed_model="gemini-embedding-001",
            ),
            # Diagram / code generation — always Flash regardless of pro mode.
            # Pro mode on diagrams adds 20-30 seconds of latency which kills
            # the conversational feel. Diagram quality is a cosmetic win that
            # isn't worth breaking the core UX.
            "gemini_diagram": GeminiProvider(
                chat_model="gemini-2.5-flash",
                embed_model="gemini-embedding-001",
            ),
            # Pro mode — stronger reasoning for the CHAT response only.
            # Used when the user explicitly toggles pro mode for better answers.
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
        """Diagram model — ALWAYS Flash, regardless of pro mode.
        
        Pro mode diagrams take 20-30s which kills the conversational UX.
        The diagram is a supporting visual, not the main event — Flash is
        good enough and keeps the product feeling snappy.
        """
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
        "Transcribe this image EXACTLY as it appears, preserving all text verbatim and "
        "describing all figures in full technical detail. This is for a voice AI assistant "
        "that will later be asked to answer questions based on this content, including "
        "questions about circuits, graphs, and diagrams. The AI cannot see the image — "
        "only your transcription — so your description must be detailed enough that "
        "someone could fully solve the problems without ever seeing the original.\n\n"
    
        "GENERAL RULES\n"
        "Do not summarize, paraphrase, or add commentary. Transcribe text verbatim. "
        "Preserve paragraph breaks, section headings, question numbering, and any marks "
        "or point values shown next to questions. If text is unclear or cut off, write "
        "[unclear] rather than guessing. Do not solve or answer any questions.\n\n"
    
        "TEXT TRANSCRIPTION\n"
        "For every piece of text on the page — titles, instructions, question numbers, "
        "question bodies, mark allocations, footnotes, headers — reproduce it word for "
        "word. Keep question numbers labeled (Q1, Q2, etc.) and preserve their original "
        "order. If there are multiple choice options, list them with their letters "
        "(a, b, c, d) and the exact option text.\n\n"
    
        "MATHEMATICAL EQUATIONS\n"
        "Write equations in plain text using standard notation. Use ^ for exponents "
        "(x^2), * for multiplication when ambiguous, / for division, sqrt() for square "
        "roots, and spell out Greek letters (omega, pi, theta, delta). For complex "
        "expressions use parentheses liberally so the order of operations is clear.\n\n"
    
        "CIRCUIT DIAGRAMS — critical detail required\n"
        "When the image contains an electrical circuit, transcribe it as a structured "
        "component list followed by a connection list. For every component, record: the "
        "component type (resistor, capacitor, voltage source, current source, inductor, "
        "etc.), its value with units (e.g., 3 ohms, 2 volts, 5 amps), its polarity or "
        "direction if shown (+ and - terminals for sources, arrow direction for current "
        "sources), and a label or reference identifier if one is given (e.g., R1, V1, Ix).\n\n"
        "Then describe the topology — which components are in series, which are in "
        "parallel, and which nodes connect to which. Use plain language like 'the 2V "
        "source is on the left side of the circuit with its positive terminal at the "
        "top, connected in series with a 3 ohm resistor which leads to the top node'. "
        "Identify any labeled currents or voltages (e.g., 'Ix flows downward through "
        "the 1 ohm resistor in the middle branch'). Be exhaustive — if the problem "
        "asks to solve for Ix using mesh analysis, the transcription must contain enough "
        "information to write the mesh equations from scratch.\n\n"
    
        "GRAPHS, WAVEFORMS, AND PLOTS\n"
        "When the image contains a graph or waveform, describe its axes, units, scale, "
        "and shape in full detail. Record: the x-axis label and range, the y-axis label "
        "and range, the peak values marked on the graph, any time period or frequency "
        "labels (e.g., T, T/2), the general shape (triangular, sinusoidal, square wave, "
        "sawtooth, exponential decay, etc.), and whether the waveform is symmetric or "
        "asymmetric about any axis. If specific points are labeled with coordinates or "
        "values, list them.\n\n"
        "For example, a triangular waveform should be described as 'a triangular "
        "waveform with peak amplitude 10V on the y-axis (labeled V), period T on the "
        "x-axis (labeled t), rising linearly from 0 to 10V between t=0 and t=T/2, then "
        "dropping linearly back to 0 by t=T, then repeating'. Enough detail to compute "
        "RMS value, form factor, or any other derived quantity.\n\n"
    
        "TABLES\n"
        "Reproduce tables row by row, column by column, preserving headers. Use a clear "
        "format like 'Row 1: column A = value, column B = value, column C = value' "
        "rather than trying to draw the table in ASCII.\n\n"
    
        "GENERAL DIAGRAMS AND FIGURES\n"
        "For any other type of figure (geometry problems, free body diagrams, block "
        "diagrams, flow charts, molecular structures, anatomical drawings), describe "
        "every labeled element, every dimension, every arrow, every angle, and every "
        "relationship shown. Include enough detail that the problem could be solved "
        "using only your description.\n\n"
    
        "OUTPUT\n"
        "Output only the transcribed and described content. No preamble, no summary, "
        "no closing remarks, no explanation of your process. Start directly with the "
        "content of the image."
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