
# Create api key
import os
import tempfile
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.types import VoiceSettings

load_dotenv()


def _get_client():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is not set")
    return ElevenLabs(api_key=api_key)


def speak(text):

    if not text or not text.strip():
        raise ValueError("speak() received empty text — nothing to convert to audio.")

    client = _get_client()

    audio = client.text_to_speech.convert(
        text=text,
        voice_id="SAz9YHcvj6GT2YYXdXww",
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.5,
            speed=0.9
        )
    )

    # Save audio to a unique temp file (avoids overwrite with concurrent users)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        filename = f.name
        for chunk in audio:
            f.write(chunk)

    # Do NOT auto-play here with os.startfile / xdg-open.
    # Gradio's gr.Audio output plays the returned file automatically
    # for the patient in the browser. That IS the "play audio" step.

    return filename


# --------------------------------------------------
# TEST CODE
# Uncomment this section when you want to test
# the voice bot independently.
# --------------------------------------------------

# text = "Hello! I am speaking automatically using ElevenLabs."
# audio_file = speak(text)
# print("Audio saved to:", audio_file)
