from typing import AsyncIterator
from Providers.open_ai import OpenAIProvider
from Providers.gemeni import GeminiProvider
from SQL.RAG import VectorRAGService


"""
Rolling summary system.
Uses the existing VectorRAGService for storage (sessions.summary column).
Character-agnostic — works with any avatar/prompt.
"""

ROLLING_SUMMARY_PROMPT = """You are a conversation summarizer. You will be given an existing summary of an ongoing conversation and new messages. Update the summary to include the new information.

Rules:
- Keep it under 4 sentences.
- Focus on topics discussed, user preferences, and key facts.
- Write in past tense.
- If the existing summary is empty, just summarize the new messages.
- Return only the updated summary, nothing else.

Existing summary:
{existing_summary}

New messages:
{new_messages}"""


class RollingSummaryManager:
    def __init__(self, gemini_provider, rag: VectorRAGService):
        """
        gemini_provider: your GeminiProvider instance (from ai.provider)
        rag: your VectorRAGService instance
        """
        self.provider = gemini_provider
        self.rag = rag
        self.counters = {}

    async def on_new_message(self, chat_id: str, recent_messages: list[dict]):
        """
        Call after every assistant response.
        Every 3rd call, regenerates the summary.
        """
        count = self.counters.get(chat_id, 0) + 1
        self.counters[chat_id] = count

        if count % 3 == 0:
            await self._update_summary(chat_id, recent_messages)

    async def _update_summary(self, chat_id: str, recent_messages: list[dict]):
        existing_summary = self.rag.get_summary(chat_id) or "No previous summary yet."

        lines = []
        for m in recent_messages:
            role = "User" if m.get("role", "").lower() == "user" else "Assistant"
            content = m.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")

        prompt = ROLLING_SUMMARY_PROMPT.format(
            existing_summary=existing_summary,
            new_messages="\n".join(lines)
        )

        try:
            updated = await self.provider.get_chat(
                system="You are a summarizer. Return only the updated summary.",
                user=prompt,
                max_output_tokens=200
            )
            updated = updated.strip()
            if updated:
                self.rag.update_summary(chat_id, updated)
        except Exception as e:
            print(f"Summary update failed: {e}")

    def build_context(self, chat_id: str) -> str:
        summary = self.rag.get_summary(chat_id)
        if not summary:
            return ""
        return f"Summary of earlier conversation: {summary}"