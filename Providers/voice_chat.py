import os
import time
import base64
import asyncio
from typing import Optional

import azure.cognitiveservices.speech as speechsdk


class VoiceChatSystem:
    def __init__(self):
        self.speech_key = (os.getenv("AZURE_SPEECH_KEY") or "").strip()
        self.speech_region = (os.getenv("AZURE_SPEECH_REGION") or "").strip()
        if not self.speech_key or not self.speech_region:
            raise RuntimeError("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION")

    def synthesize_to_ws_queue(
        self,
        loop: asyncio.AbstractEventLoop,
        out_q: "asyncio.Queue[dict]",
        text: str,
        voice_name: str,
        *,
        mp3_format: speechsdk.SpeechSynthesisOutputFormat = speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3,
        chunk_size: int = 4096,
    ) -> None:
        """
        Runs Azure TTS in a normal thread and pushes JSON messages into an asyncio.Queue.

        Messages pushed:
          - {"type":"audio","seq":int,"data_b64":str}
          - {"type":"viseme","t_ms":int,"viseme_id":int}
          - {"type":"done"}
          - {"type":"error","message":str}
        """
        try:
            speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
            speech_config.speech_synthesis_voice_name = voice_name
            speech_config.set_speech_synthesis_output_format(mp3_format)

            pull_stream = speechsdk.audio.PullAudioOutputStream()
            audio_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)

            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

            def viseme_cb(evt: speechsdk.SpeechSynthesisVisemeEventArgs):
                t_ms = int(evt.audio_offset / 10_000)  # 100ns ticks -> ms
                msg = {"type": "viseme", "t_ms": t_ms, "viseme_id": int(evt.viseme_id)}
                loop.call_soon_threadsafe(out_q.put_nowait, msg)

            synthesizer.viseme_received.connect(viseme_cb)

            fut = synthesizer.start_speaking_text_async(text)

            seq = 0
            buf = bytearray(chunk_size)

            # Pull bytes while synthesis runs
            while not fut.done():
                n = pull_stream.read(buf)
                if n and n > 0:
                    seq += 1
                    b64 = base64.b64encode(bytes(buf[:n])).decode("utf-8")
                    loop.call_soon_threadsafe(out_q.put_nowait, {"type": "audio", "seq": seq, "data_b64": b64})
                else:
                    time.sleep(0.005)

            result = fut.get()

            # Drain remaining bytes
            while True:
                n = pull_stream.read(buf)
                if not n or n <= 0:
                    break
                seq += 1
                b64 = base64.b64encode(bytes(buf[:n])).decode("utf-8")
                loop.call_soon_threadsafe(out_q.put_nowait, {"type": "audio", "seq": seq, "data_b64": b64})

            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                details = result.cancellation_details
                raise RuntimeError(f"TTS canceled: {details.reason} | {details.error_details}")

            loop.call_soon_threadsafe(out_q.put_nowait, {"type": "done"})

        except Exception as e:
            loop.call_soon_threadsafe(out_q.put_nowait, {"type": "error", "message": str(e)})
