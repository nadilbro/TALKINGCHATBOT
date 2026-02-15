import os
import time
import azure.cognitiveservices.speech as speechsdk


class VoiceChatSystem:
    def __init__(self):
        self.speech_key = (os.getenv("AZURE_SPEECH_KEY") or "").strip()
        self.speech_region = (os.getenv("AZURE_SPEECH_REGION") or "").strip()
        if not self.speech_key or not self.speech_region:
            raise RuntimeError("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION")

    def synthesize_stream_mp3(self, text: str, voice_name: str):
        """
        Generator that yields MP3 audio bytes progressively.

        Notes:
        - Uses PullAudioOutputStream so the server can start sending bytes before synthesis completes.
        - MP3 is used because WAV/PCM is awkward to stream (header needs total length).
        """
        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
        speech_config.speech_synthesis_voice_name = voice_name

        # Best for streaming over HTTP
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )

        pull_stream = speechsdk.audio.PullAudioOutputStream()
        audio_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        # Start synthesis without blocking immediately
        fut = synthesizer.start_speaking_text_async(text)

        buf = bytearray(4096)

        # Pull while running
        while not fut.done():
            n = pull_stream.read(buf)
            if n and n > 0:
                yield bytes(buf[:n])
            else:
                # avoid a tight spin if no bytes are ready yet
                time.sleep(0.005)

        # Final result + error handling
        result = fut.get()

        # Drain remaining bytes
        while True:
            n = pull_stream.read(buf)
            if not n or n <= 0:
                break
            yield bytes(buf[:n])

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = result.cancellation_details
            raise RuntimeError(f"TTS canceled: {details.reason} | {details.error_details}")
