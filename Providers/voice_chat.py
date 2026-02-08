import os
import json
import azure.cognitiveservices.speech as speechsdk

class VoiceChatSystem:
    def __init__(self):
        self.speech_key = (os.getenv("AZURE_SPEECH_KEY") or "").strip()
        self.speech_region = (os.getenv("AZURE_SPEECH_REGION") or "").strip()
        if not self.speech_key or not self.speech_region:
            raise RuntimeError("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION")


    def synthesize(self, text: str, audioName: str) -> tuple[bytes, list[dict]]:
        visemes: list[dict] = []

        def viseme_cb(evt: speechsdk.SpeechSynthesisVisemeEventArgs):
            ms = int(evt.audio_offset / 10_000) # ticks (100ns) -> ms
            visemes.append({"t_ms": ms, "viseme_id": evt.viseme_id})

        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
        speech_config.speech_synthesis_voice_name = audioName #"en-US-AvaMultilingualNeural"

        # IMPORTANT: no AudioOutputConfig => audio stays in the result
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None
        )

        synthesizer.viseme_received.connect(viseme_cb)

        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Get wav bytes
            stream = speechsdk.AudioDataStream(result)
            audio_bytes = stream.read_data(stream.get_length())
            return audio_bytes, visemes

        details = result.cancellation_details
        raise RuntimeError(f"TTS canceled: {details.reason} | {details.error_details}")
        
