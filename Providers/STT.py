import os

try:
    from deepgram import DeepgramClient
    DEEPGRAM_AVAILABLE = True
except ImportError as e:
    print(f"==> Deepgram import failed: {e}")
    DEEPGRAM_AVAILABLE = False

class DeepgramProvider:
    def __init__(self):
        if not DEEPGRAM_AVAILABLE:
            raise RuntimeError("deepgram-sdk not installed")
        
        import deepgram
        print(f"==> Deepgram SDK version: {deepgram.__version__}")
        
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")
        self.deepgram = DeepgramClient(api_key=api_key)

    def get_transcript(self, audio_bytes, mimetype="audio/webm") -> str:
        try:
            print(f"==> Deepgram: sending {len(audio_bytes)} bytes, mimetype={mimetype}")
            response = self.deepgram.listen.prerecorded.v("1").transcribe_file(
                {"buffer": audio_bytes, "mimetype": mimetype},
                model="nova-3",
                language="en",
                smart_format=True,
            )
            transcript = response.results.channels[0].alternatives[0].transcript
            print(f"==> Transcript: '{transcript}'")
            return transcript or ""
        except Exception as e:
            print(f"==> Deepgram exception: {e}")
            return ""