# AI Skin Doctor — Voice + Vision

Voice-first skin triage assistant. Speak your concern, optionally add a skin photo or short video, and get a short spoken guidance back. Turn-based (not real-time): record → transcribe → vision brain → text-to-speech → auto-play.

> AI guidance is not a medical diagnosis. Consult a licensed dermatologist for severe, sudden, or worsening symptoms.

## How it works

1. Patient speaks (Gradio mic/upload) + optional photo/video
2. Audio → text via ElevenLabs Scribe (`scribe_v2`)
3. Text + visuals → `gpt-4o-mini` vision brain (max 3 video frames, downscaled to 768px, JPEG q80)
4. Brain replies in 2–3 short TTS-safe sentences, plain English, no Markdown
5. Reply → speech via ElevenLabs TTS (`eleven_multilingual_v2`), auto-plays in browser
6. UI shows transcript, guidance text, voice player, and per-stage timing (Scribe | Brain | TTS | Total)

Safety rules (in `main_brain.py`): a single photo alone is never enough to diagnose — the bot says what it sees, states uncertainty, asks for a short video + itching/pain/duration/spreading, and always advises consulting a dermatologist. No hallucinated details.

## Quickstart

Requires Python ≥3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then fill in your keys
uv run main.py
```

Open the Gradio URL, record or upload voice, optionally add photo/video, hit **Analyze Concern**.

## Env

```
OPENAI_API_KEY=your-openai-key-here
ELEVENLABS_API_KEY=your-elevenlabs-key-here
```

`.env` is gitignored — never commit real keys.

## Project structure

- `main.py` — pipeline (`process_inputs`) + clinical Gradio Blocks UI (Patient Input | Doctor Response)
- `main_brain.py` — OpenAI vision brain, safety prompt, frame extraction, TTS cleanup
- `voice_recorder_bot.py` — ElevenLabs speech-to-text (`transcribe_voice`)
- `voice_bot.py` — ElevenLabs text-to-speech (`speak`, unique temp MP3 per request)

## Latency

Each run prints `Transcription | Brain | TTS | Total` in console and UI. Video cost dominates — capped at 3 frames. For true real-time voice-to-voice you'd need streaming STT + streaming LLM + streaming TTS over WebRTC (currently turn-based).

## Limitations

- Not a diagnosis tool; triage guidance only
- Needs clear close-up photo/video + symptom details for anything useful
- Turn-based, ~seconds per stage depending on API + media size
