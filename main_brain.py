
import os
import base64
import re
import cv2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# Creating client instance
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# Voice-first doctor: short, conversational, TTS-safe, never over-diagnoses.
# Safety-first: one photo alone is NEVER enough for a diagnosis.
SYSTEM_PROMPT = (
    "You are an AI skin-analysis assistant speaking to a patient through voice. "
    "Reply in plain conversational English with 2 to 3 short sentences only. "
    "No Markdown. No numbered lists. No bullet points. No bold. No asterisks. "
    "No hash symbols. No long disclaimer. "
    "Never claim or imply a definitive diagnosis, especially from a single photo alone. "
    "A single photo alone is never enough to diagnose. Do not hallucinate details "
    "you cannot clearly see. Use cautious words like could be or I can see. "
    "If ONLY one photo was provided and no video: describe briefly what you observe, "
    "say clearly that one photo is not enough to assess reliably, "
    "and ask for a short video plus whether it itches, hurts, how long it has been there, "
    "and whether it is spreading. End by advising to consult a dermatologist. "
    "If a video or clearer evidence plus symptoms were provided: you may mention one or two "
    "possibilities with could be, keep the uncertainty such as I cannot confirm this without an in-person exam, "
    "give one safe next step, and advise to consult a dermatologist. "
    "If no image or video was provided: ask for a clear close-up photo or a short video "
    "plus the same symptom details, and advise to consult a dermatologist. "
    "Every reply must naturally advise consulting a dermatologist in the final sentence."
)


def _clean_for_tts(text):
    """Strip Markdown/list formatting so ElevenLabs reads natural speech."""
    if not text:
        return text
    # Remove bold/italic markers, headers, bullets
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    # Collapse newlines/lines into flowing sentences
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def brain_of_the_doctor(patient_text, image_filepath=None, video_filepath=None):

    if not patient_text or not patient_text.strip():
        raise ValueError("patient_text is empty — transcription failed or no audio provided.")

    has_image = bool(image_filepath)
    has_video = bool(video_filepath)

    if has_image and not has_video:
        evidence_note = (
            "Evidence: a single photo only, no video. "
            "Follow the single-photo rule: do not diagnose, ask for a short video "
            "and symptom details, and advise consulting a dermatologist."
        )
    elif has_video:
        evidence_note = (
            "Evidence: video frames provided (possibly with a photo). "
            "You may discuss possibilities cautiously with uncertainty, "
            "and still advise consulting a dermatologist."
        )
    else:
        evidence_note = (
            "Evidence: no image or video provided, only the patient's words. "
            "Ask for a clear photo or short video plus symptom details, "
            "and advise consulting a dermatologist."
        )

    # Creating content
    content = [
        {
            "type": "text",
            "text": f"{patient_text.strip()}\n\n[{evidence_note}]"
        }
    ]

    MAX_FRAMES = 3
    MAX_SIDE_PX = 768


    # If video is provided
    if video_filepath:

        cap = cv2.VideoCapture(video_filepath)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_filepath}")

        try:
            # Count FPS and frames
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if not fps or fps <= 0 or frame_count <= 0:
                # Fallback: treat as a single-frame video
                duration = 1.0
                frame_count = 1
            else:
                duration = frame_count / fps
                print("Duration:", duration, "seconds")

            # Calculate number of frames to extract (capped to avoid token blowup)
            frames_per_second = 1
            number_of_frames = max(
                1,
                min(MAX_FRAMES, int(duration * frames_per_second))
            )

            print("Frames to extract:", number_of_frames)

            # Extract frames
            frames = []

            if number_of_frames == 1:
                target_frames = [0]
            else:
                target_frames = [
                    int(
                        i * (frame_count - 1)
                        / (number_of_frames - 1)
                    )
                    for i in range(number_of_frames)
                ]

            print("Target frame positions:", target_frames)

            current_frame = 0

            while True:

                success, frame = cap.read()

                if not success:
                    break

                if current_frame in target_frames:
                    frames.append(frame)
                    print(
                        f"Frame {current_frame} extracted successfully"
                    )

                current_frame += 1

            print(
                "Frames successfully extracted:",
                len(frames)
            )

            # Convert frames to Base64 (downscaled to keep vision fast)
            base64_frames = []

            for frame in frames:

                h, w = frame.shape[:2]
                scale = min(1.0, MAX_SIDE_PX / max(h, w))
                if scale < 1.0:
                    frame = cv2.resize(
                        frame,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA
                    )

                success, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )

                if success:
                    frame_data = base64.b64encode(
                        buffer
                    ).decode("utf-8")

                    base64_frames.append(frame_data)

            print(
                "Frames converted to Base64:",
                len(base64_frames)
            )
        finally:
            cap.release()

        # Add every extracted frame
        for frame_data in base64_frames:

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_data}"
                    }
                }
            )


    # If image is provided (independent of video — both can be sent)
    if image_filepath:

        with open(image_filepath, "rb") as image_file:

            image_data = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            }
        )


    # Put everything into messages with a voice-first doctor system prompt
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": content
        }
    ]


    # Sending request to OpenAI (short output = faster + TTS-friendly + less hallucination)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=150,
        temperature=0.2,
        messages=messages
    )


    # Return doctor's response, cleaned for TTS
    return _clean_for_tts(response.choices[0].message.content)


# --------------------------------------------------
# TEST CODE
# Uncomment this section when you want to test
# the function independently.
# --------------------------------------------------

# response = brain_of_the_doctor(
#     patient_text="I have redness and itching on my hand.",
#     image_filepath="skin.jpg",
#     video_filepath="hand_skin_demo.mp4"
# )

# print(response)
