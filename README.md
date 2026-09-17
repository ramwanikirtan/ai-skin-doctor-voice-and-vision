# AI Skin Doctor — Voice + Vision

A voice-first skin triage assistant. You **speak** your concern, optionally add a **skin photo** or **short video**, and a clinical AI brain replies in **short spoken guidance** that auto-plays in the browser.

> ⚠️ AI guidance is not a medical diagnosis. Always consult a licensed dermatologist for severe, sudden, or worsening symptoms.

---

## 1. What it is

Most symptom checkers are text forms with generic output. This project explores a different interaction: **talk like you would to a doctor, show what you see, hear back what to do next**.

Concretely, the app is a Gradio web workspace with two panels:

- **Patient Input (left):** microphone/upload voice recording, skin photo upload, optional short video rotation, and an **Analyze Concern** button.
- **Doctor Response (right):** the transcribed patient speech, the doctor's guidance text (2–3 sentences), an auto-playing voice reply, per-stage pipeline timing, and Download Summary / Forward to Doctor actions.

It is deliberately **turn-based, not real-time**: one full pass is record → transcribe → think → speak → play. That keeps the architecture simple and debuggable while surfacing the real engineering problems (prompt safety, vision cost, latency breakdown) before investing in streaming infrastructure.

### What it is NOT

- Not a diagnostic device and never claims a definitive diagnosis.
- Not real-time voice-to-voice (no duplex streaming, no barge-in — see Future changes).
- Not a replacement for a dermatologist; every reply advises consulting one.

---

## 2. How it works — the pipeline

```
┌──────────┐   ┌───────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌───────────┐
│ 1. Speak │ → │ 2. Transcribe │ → │ 3–4. Vision      │ → │ 5. Speak     │ → │ 6. Listen │
│  mic /   │   │  ElevenLabs   │   │  brain           │   │  ElevenLabs  │   │  browser  │
│  upload  │   │  Scribe v2    │   │  gpt-4o-mini     │   │  TTS         │   │  autoplay │
│  + photo │   │  audio → text │   │  text+frames →   │   │  text → mp3  │   │  gr.Audio │
│  + video │   │               │   │  2–3 sentences   │   │              │   │           │
└──────────┘   └───────────────┘   └──────────────────┘   └──────────────┘   └───────────┘
        each arrow timed: Transcription Xs | Brain Xs | TTS Xs | Total Xs
```

**Step 1 — Patient speaks and shows.** `main.py:process_inputs()` receives three Gradio inputs: `audio_filepath` (mic or upload), `image_filepath` (photo), `video_filepath` (short rotation clip). Photo and video sit side-by-side so the whole form fits one screen.

**Step 2 — Audio → text.** `voice_recorder_bot.py:transcribe_voice()` sends the audio file to ElevenLabs speech-to-text (`scribe_v2`) and returns `patient_text`. Missing API key or missing file raises a clear `ValueError`/`FileNotFoundError` instead of crashing the UI.

**Step 3 — Text + visuals → brain.** `main_brain.py:brain_of_the_doctor()` builds one multimodal user message:

- The transcribed patient words, plus an **evidence note** the code injects (`single photo only` / `video frames provided` / `no image or video`) so the model always knows which situation it is in.
- Up to **3 video frames** sampled at ~1 fps (downscaled to max 768px side, JPEG quality 80, base64 data-URLs) — capped so a 10-second clip doesn't become 10 full-size images.
- The optional photo as a base64 image (independent of video — both can be sent together).
- A strict system prompt (see Safety below) with `temperature=0.2`, `max_tokens=150` for short, low-hallucination, TTS-friendly output.

**Step 4 — Brain → short text.** The model returns 2–3 conversational sentences. A `_clean_for_tts()` post-processor strips any leftover Markdown (`**`, `#`, `-`, `1.`) and collapses newlines so ElevenLabs reads natural speech instead of "asterisk asterisk eczema".

**Step 5 — Text → voice.** `voice_bot.py:speak()` converts the reply with ElevenLabs TTS (`eleven_multilingual_v2`, `VoiceSettings(stability=0.5, speed=0.9)`) and saves it to a **unique temp MP3 per request** (no shared `voice.mp3` overwrite between users). No `os.startfile`/`xdg-open` server-side playback — that was the old `play_audio` crash.

**Step 6 — Play for the patient.** The MP3 path is returned into `gr.Audio(autoplay=True)`, so the browser plays it. Errors at any stage are caught in `process_inputs()` and shown as friendly text plus a timing note instead of a red Gradio traceback.

---

## 3. Safety + uncertainty design

This is the most important part of the project. A skin bot that names a disease from one blurry photo is dangerous and unimpressive. The design instead:

1. **One photo alone is never enough.** The system prompt states this explicitly, and the code reinforces it with the injected evidence note. Photo-only replies must describe briefly, state that one photo can't be assessed reliably, ask for a short video plus itching/pain/duration/spreading, and advise seeing a dermatologist.
2. **Video unlocks cautious discussion.** Only when video frames (or photo + symptom details) exist may the model mention one or two possibilities — always hedged (`could be`, `consistent with`, `I cannot confirm this without an in-person exam`).
3. **No evidence → ask, don't guess.** Text-only input triggers a request for a clear close-up photo or short video plus the same symptom details.
4. **Dermatologist in every reply.** The final sentence always advises consulting a dermatologist, phrased conversationally (short, no long legal disclaimer that TTS would read out).
5. **Low randomness.** `temperature=0.2` and a 150-token cap reduce rambling and hallucinated details.

Example of the intended tone:

> "The redness, dryness, and peeling could be related to eczema, irritation, or contact dermatitis. I can't reliably tell which one from this image alone, so a short video or more information about itching, pain, and how long you've had it would help, and please consult a dermatologist."

---

## 4. Tech stack

| Layer        | Technology | Details |
|---|---|---|
| UI           | Gradio 6.x Blocks | Custom "Clinical Clarity" CSS (azure `#0284C7` / teal `#0D9488`, Plus Jakarta Sans + Inter), 7/5 workspace grid, header + footer |
| Speech→text  | ElevenLabs Scribe | `speech_to_text.convert(..., model_id="scribe_v2")` |
| Vision brain | OpenAI `gpt-4o-mini` | Multimodal chat, `temperature=0.2`, `max_tokens=150`, base64 frames |
| Text→speech  | ElevenLabs TTS | `text_to_speech.convert(..., model_id="eleven_multilingual_v2")`, unique temp MP3 |
| Vision utils | OpenCV, base64 | Frame sampling, resize, JPEG encode |
| Config       | python-dotenv, `uv` | `.env` keys, `uv sync` / `uv run` |
| Language     | Python ≥3.13 | `main.py`, `main_brain.py`, `voice_bot.py`, `voice_recorder_bot.py` |

---

## 5. How to use

### Setup

```bash
uv sync
cp .env.example .env   # then edit .env with your keys
```

`.env` contents (never commit real keys — `.env` is gitignored):

```
OPENAI_API_KEY=your-openai-key-here
ELEVENLABS_API_KEY=your-elevenlabs-key-here
```

### Run

```bash
uv run main.py
```

Open the printed Gradio URL (local or share link).

### Consultation flow

1. **Describe your skin concern** — Record (mic) or upload audio. Mention duration, sensation (itch/pain), and triggers.
2. **Skin photo & video** — Upload a clear close-up (natural light, 10–15 cm). A short rotation video is optional but helps a lot.
3. Press **Analyze Concern** — watch the four outputs fill in: transcript, guidance text, voice player (auto-plays), timing line.
4. **Download Summary** saves a `.txt` (transcript + guidance + timing + notice) for your records; **Forward to Doctor** reminds you to share it.
5. **Clear Form** resets all inputs/outputs for the next session.

### Suggested test matrix

- Clear close-up photo + detailed voice → cautious possibilities + next step
- Blurry/single photo + vague voice → ask for video + symptoms, no disease name
- Voice only (no media) → ask for photo/video
- Short video (2–5s) vs longer video (10s+) → compare timing line and answer quality

---

## 6. Project structure

```
main.py                # pipeline process_inputs() + Gradio Blocks clinical UI
main_brain.py          # SYSTEM_PROMPT, _clean_for_tts(), frame extraction, OpenAI call
voice_recorder_bot.py  # transcribe_voice() (ElevenLabs Scribe) + legacy mic recorder
voice_bot.py           # speak() (ElevenLabs TTS → unique temp MP3)
.env.example           # placeholder keys (copy to .env)
pyproject.toml / uv.lock
hand_skin_demo.mp4 / first_frame.jpg / open-hand-...jpg  # sample media
```

---

## 7. Latency — where the ~23s goes

Every run logs and displays:

```
Transcription: Xs | Brain: Xs | TTS: Xs | Total: Xs
```

Measured stages: Scribe STT, OpenAI vision (frames + tokens), ElevenLabs TTS. Known cost drivers: number/size of video frames sent to vision, output token length (capped at 150 for this reason), and TTS synthesis of the reply. Current mitigations: 3-frame cap, 768px downscale, short outputs. Streaming (below) is the real fix.

---

## 8. What I learned

- **Prompt engineering is safety engineering.** The evidence-note injection (telling the model which media it actually got) was more reliable than trusting the system prompt alone.
- **Design for the output modality.** TTS punishes Markdown, lists, and disclaimers that look fine as text — the 2–3 sentence plain-English constraint exists because of the speaker, not the screen.
- **Vision cost is frame cost.** Uncapped 1-fps sampling turns a 10s video into 10 images; capping + downscaling is the cheapest latency win.
- **Server-side audio playback is a trap.** `os.startfile`/media players have no place in a web backend — return the file and let the browser play it.
- **Measure before optimizing.** Per-stage timing turned a vague "it's slow" into an assignable bottleneck.
- **Medical UX needs calibrated uncertainty.** Saying "I can't tell from this photo, here's what would help" is the correct answer most of the time.

---

## 9. Future changes

### 🧠 Improve AI accuracy & uncertainty
- Confidence-gated replies (answer vs ask-for-more threshold), photo quality scoring (blur/lighting/distance check before calling vision), and structured visual observations the UI can render as cards.

### ⚡ Reduce 23s latency
- Stream the pipeline: partial transcripts → `stream=True` LLM tokens → chunked TTS playback, plus parallelize independent work and cache repeated syntheses.

### 🎥 Smarter video/frame processing
- Sharpest-frame selection instead of evenly spaced frames, adaptive count by clip length/motion, lesion cropping/zoom, and lighting normalization before encoding.

### 🛡️ Safety + input validation
- File type/size/duration guards on all uploads, PII warning, age-gated wording, refusal paths for non-skin or emergency red-flag symptoms, and audit logging of evidence notes.

### 📊 Evaluation + metrics
- Golden test set (clear / ambiguous / no-image / short / long video / voice-only) with latency-vs-quality tracking per run, prompt regression diffs, and human rating of uncertainty calibration.

### 🧪 Automated tests
- Unit tests for `_clean_for_tts`, evidence-note branching, frame extraction (mocked video), and `process_inputs` ordering with mocked APIs; CI running syntax + mock-pipeline checks on every push.

---

## 10. Limitations & disclaimer

Turn-based (not real-time), English-first, quality depends on media/lighting and API availability/cost. This is a learning project, not a medical device. **AI guidance is not a medical diagnosis — consult a licensed doctor for severe, sudden, or worsening symptoms.**
