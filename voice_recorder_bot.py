# Recording users voice

import os
from dotenv import load_dotenv
import logging
import speech_recognition as sr
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from io import BytesIO

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def record_voice(file_path, timeout=20, phrase_time_limit=None):

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        logging.info("Adjusting for ambient noise...")

        recognizer.adjust_for_ambient_noise(
            source,
            duration=1
        )

        logging.info("Listening for speech...")

        # Record the audio
        audio = recognizer.listen(
            source,
            timeout=timeout,
            phrase_time_limit=phrase_time_limit
        )

        # Convert the audio to wav format
        wav_data = audio.get_wav_data()

        audio_segment = AudioSegment.from_wav(
            BytesIO(wav_data)
        )

        audio_segment.export(
            file_path,
            format="mp3",
            bitrate="128k"
        )

        logging.info(
            "Audio saved to %s",
            file_path
        )

        return file_path


def transcribe_voice(audio_filepath):

    api_key = os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        raise ValueError(
            "ELEVENLABS_API_KEY is not set"
        )

    if not audio_filepath or not os.path.exists(audio_filepath):
        raise FileNotFoundError(
            f"Audio file not found: {audio_filepath}"
        )

    client = ElevenLabs(
        api_key=api_key
    )

    # Open recorded audio file
    with open(audio_filepath, "rb") as audio_file:

        # Send audio to ElevenLabs Speech-to-Text
        transcription = client.speech_to_text.convert(
            file=audio_file,
            model_id="scribe_v2"
        )

    # Return the transcribed text
    return transcription.text


# --------------------------------------------------
# TEST CODE
# Uncomment this section when you want to test
# the voice recorder and transcription independently.
# --------------------------------------------------

# audio_file_path = "user_voice.mp3"

# record_voice(
#     audio_file_path,
#     timeout=20,
#     phrase_time_limit=10
# )

# text = transcribe_voice(audio_file_path)

# print("User said:", text)
