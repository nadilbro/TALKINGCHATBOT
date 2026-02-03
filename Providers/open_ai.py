from typing import AsyncIterator
from openai import AsyncOpenAI

class OpenAIProvider:
    def __init__(self, chat_model: str, embed_model: str):
        self.client = AsyncOpenAI()
        self.chat_model = chat_model
        self.embed_model = embed_model

    async def embed(self, text: str) -> list[float]:
        resp = await self.client.embeddings.create(model=self.embed_model, input=text)
        return resp.data[0].embedding

    async def _stream(self, system: str, user: str, max_output_tokens: int = 120) -> AsyncIterator[str]:
        stream = await self.client.responses.create(
            model=self.chat_model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
            max_output_tokens=max_output_tokens,
        )

        full = ""
        got_delta = False

        async for event in stream:
            t = getattr(event, "type", None)
            print("EVENT TYPE:", t)

            if t == "response.output_text.delta":
                got_delta = True
                full += event.delta
                yield event.delta

            elif t == "response.completed":
                # If no deltas were streamed, flush final output_text once
                if not got_delta:
                    final_text = getattr(getattr(event, "response", None), "output_text", "") or ""
                    if final_text:
                        yield final_text
                    elif full:
                        yield full
                return

            elif t in ("response.failed", "response.incomplete"):
                # best effort flush
                if not got_delta and full:
                    yield full
                return

    def stream_chat(
        self,
        system: str,
        user: str,
        max_output_tokens: int = 120
    ) -> AsyncIterator[str]:
        return self._stream(system=system, user=user, max_output_tokens=max_output_tokens)
    