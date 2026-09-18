
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


# Voice-first doctor: uncertainty-aware, evidence-driven, never templated.
# The reply adapts to the case: what was asked, what is visible, what the
# references support, and what is genuinely still needed. Safety without a
# fixed script: no mandatory video request, no identical disclaimer each time.
SYSTEM_PROMPT = (
    "You are an AI skin-analysis assistant speaking to a patient through voice. "
    "Reply in plain conversational English with no Markdown, no lists, no bullet points, "
    "no bold, no asterisks, no hash symbols — the reply will be read aloud. "
    "Base your answer on three things together: what the patient asked, what is actually "
    "visible in any photo or video frames, and the reference passages provided below. "
    "Describe only what you can actually see; never invent visual details. "
    "Never claim a definitive diagnosis from an image. Where uncertainty fits, use cautious "
    "language such as could be consistent with, one possibility is, the appearance may fit, "
    "or this cannot be confirmed from an image alone. "
    "When a reference passage is genuinely relevant, use it: connect what you observe to the "
    "passage's general medical information, for example According to the American Academy "
    "of Dermatology, itching and redness are common features of eczema. "
    "Ignore passages that are irrelevant to this case — never force them into the answer. "
    "Never present a retrieved passage as proof of this patient's condition, and never invent "
    "sources or citations beyond the passages given. "
    "Clearly keep apart what you observe, what it could be and why, what the references add, "
    "and what remains uncertain. "
    "Decide yourself what is needed next. If the image and question already allow a useful "
    "explanation, give it and do not demand a video. Ask for a clearer photo, a short video, "
    "or symptom details such as onset, itch or pain, spreading, or new products touching the "
    "area only when that information would genuinely improve the assessment. "
    "Recommend prompt in-person care when warning signs are present, such as rapidly spreading "
    "redness, swelling, pus or discharge, red streaking, severe pain, or fever with a rash. "
    "Otherwise word any care-seeking note naturally for the case instead of repeating one fixed "
    "disclaimer. "
    "Let length and structure follow the case: a simple question gets a concise answer, while a "
    "complex image earns a fuller explanation covering what you see, what it could be and why, "
    "what the references add, what may help, and what further information would help. "
    "End your reply with one machine-readable line naming the sources you actually relied on: "
    "USED_SOURCES: <comma-separated source numbers, or the word none>. "
    "That line is removed before the patient hears anything, so always include it."
)


def _build_evidence_block(evidence):
    """Format retrieved passages as structured, citable sources for the LLM."""
    if not evidence:
        return ""
    parts = []
    for i, chunk in enumerate(evidence, start=1):
        parts.append(
            f"SOURCE {i}\n"
            f"Organization: {chunk.get('source_org', 'unknown source')}\n"
            f"Title: {chunk.get('title', '')}\n"
            f"Section: {chunk.get('section', '')}\n"
            f"Content: {chunk.get('text', '')}"
        )
    return (
        "Reference passages — use the ones relevant to this case, ignore the rest:\n\n"
        + "\n\n".join(parts)
    )


def _source_names(evidence):
    """Deduped organization names for UI display (never sent to TTS)."""
    names = []
    for chunk in evidence or []:
        org = (chunk.get("source_org") or "").strip()
        if org and org not in names:
            names.append(org)
    return names


def _used_sources(reply_text, evidence):
    """Parse the model's USED_SOURCES trailer into 'Org — Title' refs.

    Falls back to all retrieved orgs if the trailer is missing/unparseable,
    so the UI never ends up with less information than before.
    Returns (cleaned_reply_text, source_refs).
    """
    refs, cleaned = _source_names(evidence), reply_text
    if not reply_text:
        return cleaned, refs
    matches = re.findall(
        r"USED_SOURCES\s*:\s*([^\n\r]+)",
        reply_text, flags=re.IGNORECASE,
    )
    if not matches:
        return cleaned, refs
    cleaned = re.sub(
        r"[ \t]*USED_SOURCES\s*:\s*[^\n\r]*",
        "", reply_text, flags=re.IGNORECASE,
    ).strip()
    trailer = matches[-1].strip().lower()
    if trailer in ("none", "n/a", "-"):
        return cleaned, []
    used = []
    for token in re.split(r"[,\s;]+", trailer):
        if token.isdigit():
            idx = int(token) - 1
            if evidence is not None and 0 <= idx < len(evidence):
                chunk = evidence[idx]
                org = (chunk.get("source_org") or "unknown source").strip()
                title = (chunk.get("title") or chunk.get("section") or "").strip()
                ref = f"{org} — {title}" if title else org
                if ref not in used:
                    used.append(ref)
    # If the model wrote numbers that match nothing, keep the safe fallback.
    return cleaned, (used or refs)


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


def brain_of_the_doctor(patient_text, image_filepath=None, video_filepath=None, evidence=None):
    """Run vision + grounded generation. Returns (response_text, source_org_names).

    evidence: list of retrieved chunk dicts from rag.retriever.retrieve()
    (may be empty/None — the brain still works, with an explicit no-evidence note).
    """

    if not patient_text or not patient_text.strip():
        raise ValueError("patient_text is empty — transcription failed or no audio provided.")

    has_image = bool(image_filepath)
    has_video = bool(video_filepath)

    if has_image and not has_video:
        evidence_note = (
            "What was provided: one photo plus the patient's words, no video. "
            "Judge the photo on its own merits — if it clearly shows the area, "
            "give a useful description and explanation without demanding a video; "
            "only ask for a video, a clearer photo, or symptom details if that "
            "information would genuinely change the assessment."
        )
    elif has_video:
        evidence_note = (
            "What was provided: video frames (possibly with a photo) plus the "
            "patient's words. Use the frames together with the words, staying "
            "uncertain where the images cannot settle the matter."
        )
    else:
        evidence_note = (
            "What was provided: only the patient's words, no image or video. "
            "Answer what you can from the words and references; ask for a photo "
            "or symptom details only if they would genuinely help."
        )

    # Creating content
    user_text = patient_text.strip() + f"\n\n[{evidence_note}]"
    evidence_block = _build_evidence_block(evidence)
    if evidence_block:
        user_text += f"\n\n{evidence_block}"
    else:
        user_text += "\n\n[No reference passages retrieved — rely on careful observation only.]"

    content = [
        {
            "type": "text",
            "text": user_text
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


    # Sending request to OpenAI (length follows the case: concise when simple,
    # fuller when the image/complaint needs it; still TTS-friendly speech)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=400,
        temperature=0.5,
        messages=messages
    )


    raw_reply = response.choices[0].message.content
    # Strip the machine-readable trailer (never spoken), attribute real sources.
    reply_no_trailer, source_refs = _used_sources(raw_reply, evidence)

    # Return (doctor's response cleaned for TTS, source refs for UI display)
    return _clean_for_tts(reply_no_trailer), source_refs


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
