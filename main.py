# Logic + UI
# Full pipeline:
# 1. User asks question in audio
# 2. Audio -> text (transcribe, timed)
# 3. Text + image/video -> brain_of_the_doctor (timed)
# 4. Brain responds in text (2-3 sentences, TTS-safe, uncertainty-aware)
# 5. Text response -> audio response (speak, timed)
# 6. Play audio for the patient (Gradio gr.Audio auto-plays returned file)

import time

import gradio as gr

# Brain
from main_brain import brain_of_the_doctor

# Voice Recorder + Transcription
from voice_recorder_bot import transcribe_voice

# Voice Bot
from voice_bot import speak


def _fmt_timing(label, seconds):
    return f"{label}: {seconds:.1f}s"


def process_inputs(audio_filepath, image_filepath, video_filepath):
    # STEP 1: user asked question in audio (audio_filepath from Gradio microphone/upload)
    if audio_filepath is None:
        return (
            "No audio provided.",
            "Please record or upload your voice question.",
            None,
            "No timing — no audio received.",
        )

    total_start = time.perf_counter()

    try:
        # STEP 2: this audio will be converted to text (ElevenLabs Scribe)
        t0 = time.perf_counter()
        patient_text = transcribe_voice(audio_filepath)
        t_transcribe = time.perf_counter() - t0

        # STEP 3: this text + user's image/video will be sent to brain of the doctor
        t0 = time.perf_counter()
        doctor_response = brain_of_the_doctor(
            patient_text=patient_text,
            image_filepath=image_filepath,
            video_filepath=video_filepath,
        )
        t_brain = time.perf_counter() - t0

        # STEP 4: brain of the doctor responded in text (doctor_response)

        # STEP 5: convert this text response from the doctor to audio response
        t0 = time.perf_counter()
        doctor_response_audio_filepath = speak(doctor_response)
        t_tts = time.perf_counter() - t0

        total = time.perf_counter() - total_start
        timing_text = (
            f"{_fmt_timing('Transcription', t_transcribe)} | "
            f"{_fmt_timing('Brain', t_brain)} | "
            f"{_fmt_timing('TTS', t_tts)} | "
            f"{_fmt_timing('Total', total)}"
        )
        print(f"[pipeline] {timing_text}")

        # STEP 6: play audio for the patient.
        # No play_audio() / os.startfile call needed — returning the filepath
        # in the gr.Audio output makes Gradio play it in the browser.
        return patient_text, doctor_response, doctor_response_audio_filepath, timing_text

    except FileNotFoundError as e:
        return "Audio file error.", str(e), None, "No timing — file error."
    except ValueError as e:
        return "Input error.", str(e), None, "No timing — input error."
    except Exception as e:
        total = time.perf_counter() - total_start
        return (
            "Error processing request.",
            f"Something went wrong: {e}",
            None,
            f"Failed after {total:.1f}s.",
        )


def _build_summary_file(speech_text, guidance_text, timing_text):
    """Write a downloadable consultation summary, return its path."""
    import tempfile

    content = (
        "AI Skin Specialist — Consultation Summary\n"
        "==========================================\n\n"
        f"Patient speech (transcribed):\n{speech_text or '-'}\n\n"
        f"Doctor's guidance:\n{guidance_text or '-'}\n\n"
        f"Pipeline timing:\n{timing_text or '-'}\n\n"
        "Medical Notice: AI guidance is not a medical diagnosis. "
        "Consult a licensed doctor for severe, sudden, or worsening symptoms.\n"
    )
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".txt", prefix="skin_summary_",
        mode="w", encoding="utf-8",
    ) as f:
        f.write(content)
        return f.name


# ---------------------------------------------------------------------------
# UI — Clinical Clarity (Stitch: stitch_ai_skin_specialist_interface)
# Matches DESIGN.md + screen.png + code.html:
# header, 7/5 workspace grid, Patient Input | Doctor Response, footer notice.
# ---------------------------------------------------------------------------

CLINICAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

.gradio-container { background: #F8FAFC !important; font-family: 'Plus Jakarta Sans', 'Inter', system-ui, sans-serif !important; }
/* Compact one-screen layout: kill Gradio's big default gaps/heights */
.gradio-container .form, .gradio-container .block { gap: 8px !important; }
footer { display: none !important; }
.clinical-header { background: rgba(255,255,255,.92); backdrop-filter: blur(12px); border-bottom: 1px solid #E2E8F0; padding: 8px 20px; display: flex; align-items: center; justify-content: space-between; gap: 12px; position: sticky; top: 0; z-index: 50; }
.clinical-brand { display: flex; align-items: center; gap: 10px; }
.clinical-logo { width: 32px; height: 32px; border-radius: 8px; background: #0284C7; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: 700; box-shadow: 0 1px 3px rgba(15,23,42,.12); }
.clinical-brand-name { font-weight: 700; font-size: 14px; color: #0F172A; }
.clinical-brand-sub { font-size: 11px; color: #64748B; }
.clinical-nav { display: flex; gap: 4px; }
.clinical-nav a { padding: 5px 10px; border-radius: 8px; font-size: 12px; color: #64748B; text-decoration: none; font-family: 'Inter', sans-serif; }
.clinical-nav a.active { background: #E0F2FE; color: #0284C7; font-weight: 600; }
.clinical-pill-btn { display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 8px; background: #F1F5F9; color: #475569; font-size: 11px; font-family: 'Inter', sans-serif; }
.clinical-workspace { max-width: 1400px; margin: 0 auto; padding: 10px 20px 12px; }
.clinical-title { font-size: 22px; font-weight: 600; letter-spacing: -0.02em; color: #0F172A; margin: 0; }
.clinical-subtitle { font-size: 13px; color: #64748B; margin: 0 0 8px; }
.clinical-card { background: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 14px !important; box-shadow: 0 1px 3px rgba(15,23,42,.04), 0 4px 12px rgba(2,132,199,.03); padding: 12px !important; }
.clinical-card-head { display: flex; align-items: center; gap: 8px; background: #F2F3FF; margin: -12px -12px 8px; padding: 8px 12px; border-radius: 14px 14px 0 0; }
.clinical-card-icon { width: 28px; height: 28px; border-radius: 7px; background: #006194; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 15px; }
.clinical-card-icon.teal { background: #006a61; }
.clinical-card-h { font-size: 14px; font-weight: 600; color: #0F172A; margin: 0; }
.clinical-card-sub { font-size: 11px; color: #64748B; margin: 0; }
.clinical-section-label { font-size: 13px; font-weight: 600; color: #0F172A; display: flex; align-items: center; gap: 6px; margin: 2px 0 4px; }
.clinical-section-label .mi { color: #0284C7; font-size: 15px; }
.clinical-hint { font-size: 11px; color: #64748B; margin: 0 0 4px; }
.clinical-input { background: #F2F3FF; border-radius: 10px; padding: 8px; }
#analyze-btn { background: #006194 !important; color: #fff !important; font-size: 14px !important; font-weight: 600 !important; border-radius: 8px !important; padding: 10px !important; margin-top: 4px !important; }
#analyze-btn:hover { background: #0369A1 !important; }
/* Shrink Gradio media drop zones so image+video fit without scrolling */
.fit-media { min-height: 0 !important; }
.fit-media [data-testid="image"], .fit-media [data-testid="video"], .fit-media [data-testid="audio"] { min-height: 0 !important; }
.fit-media .image-container, .fit-media .video-container { min-height: 120px !important; max-height: 170px !important; }
.fit-media img, .fit-media video { max-height: 150px !important; object-fit: contain !important; }
.fit-media .wrap { min-height: 0 !important; }
.clinical-card textarea { min-height: 38px !important; font-size: 13px !important; }
.clinical-output-speech textarea { font-style: italic !important; }
.clinical-footer { background: #fff; border-top: 1px solid #E2E8F0; padding: 8px 20px; text-align: center; font-size: 11px; color: #64748B; }
.clinical-footer strong { color: #0F172A; }
.clinical-timing textarea { font-family: 'Inter', monospace !important; font-size: 11px !important; }
.clinical-card label { font-size: 12px !important; margin-bottom: 2px !important; }
@media (max-width: 768px) { .clinical-workspace { padding: 10px; } .clinical-nav { display: none; } }
"""

HEADER_HTML = """
<div class="clinical-header">
  <div class="clinical-brand">
    <div class="clinical-logo">✚</div>
    <div>
      <div class="clinical-brand-name">AI Skin Specialist</div>
      <div class="clinical-brand-sub">Voice, image, and video based skin consultation assistant</div>
    </div>
  </div>
  <nav class="clinical-nav">
    <a href="#" class="active">Consultation</a>
    <a href="#">Patient History</a>
    <a href="#">Clinical Guidelines</a>
  </nav>
  <div style="display:flex;gap:8px;align-items:center;">
    <span class="clinical-pill-btn">⟳ New Session</span>
    <span class="clinical-pill-btn">🔊 Audio Guide</span>
  </div>
</div>
"""

TITLE_HTML = """
<div style="margin-bottom:4px;">
  <h1 class="clinical-title">Dermatological Assessment Workspace</h1>
  <p class="clinical-subtitle">Upload or record your symptoms for clinical-grade guidance and triage support.</p>
</div>
"""

FOOTER_HTML = """
<div class="clinical-footer">🛡 <strong>Medical Notice:</strong> AI guidance is not a medical diagnosis.
Consult a licensed doctor for severe, sudden, or worsening symptoms.</div>
"""


with gr.Blocks(title="AI Skin Specialist — Dermatological Assessment Workspace") as demo:
    gr.HTML(HEADER_HTML)
    with gr.Column(elem_classes=["clinical-workspace"]):
        gr.HTML(TITLE_HTML)
        with gr.Row(equal_height=False):
            # LEFT — Patient Input (7 cols)
            with gr.Column(scale=7, elem_classes=["clinical-card"]):
                gr.HTML(
                    '<div class="clinical-card-head"><div class="clinical-card-icon">🩺</div>'
                    '<div><h2 class="clinical-card-h">Patient Input</h2>'
                    '<p class="clinical-card-sub">Provide vocal descriptions and clear close-up imagery</p></div></div>'
                )
                gr.HTML('<div class="clinical-section-label"><span class="mi">🎙</span>Describe your skin concern</div>')
                with gr.Column(elem_classes=["clinical-input"]):
                    audio_in = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Voice (duration, sensation, triggers)",
                        elem_classes=["fit-media"],
                    )
                gr.HTML('<div class="clinical-section-label"><span class="mi">📷🎥</span>Skin photo & video <span class="clinical-hint">(video optional — helps a lot)</span></div>')
                gr.HTML('<div class="clinical-hint">Natural light, 10–15 cm from affected area.</div>')
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=0):
                        image_in = gr.Image(
                            type="filepath",
                            label="Clear photo",
                            height=140,
                            elem_classes=["fit-media"],
                        )
                    with gr.Column(scale=1, min_width=0):
                        video_in = gr.Video(
                            label="Video rotation",
                            height=140,
                            elem_classes=["fit-media"],
                        )
                analyze_btn = gr.Button("✚  Analyze Concern  →", elem_id="analyze-btn", variant="primary")
                with gr.Row():
                    clear_btn = gr.Button("⟳ Clear Form", variant="secondary", size="sm")
                    gr.HTML(
                        '<div class="clinical-hint" style="text-align:right;">Examples: '
                        '<b style="color:#0284C7;">Contact Dermatitis</b> · Eczema</div>'
                    )

            # RIGHT — Doctor Response (5 cols)
            with gr.Column(scale=5, elem_classes=["clinical-card"]):
                gr.HTML(
                    '<div class="clinical-card-head"><div class="clinical-card-icon teal">🤖</div>'
                    '<div><h2 class="clinical-card-h">Doctor Response</h2>'
                    '<p class="clinical-card-sub">Consultation Assessment</p></div></div>'
                )
                gr.HTML('<div class="clinical-section-label"><span class="mi">🗣</span>Your speech</div>')
                speech_out = gr.Textbox(
                    label="Transcribed patient speech",
                    placeholder="Your transcribed speech will appear here…",
                    lines=2,
                    max_lines=3,
                    elem_classes=["clinical-output-speech"],
                )
                gr.HTML('<div class="clinical-section-label"><span class="mi">⚕</span>Doctor\'s guidance</div>')
                guidance_out = gr.Textbox(
                    label="Guidance (2–3 sentences, voice-friendly)",
                    placeholder="Doctor guidance will appear here…",
                    lines=3,
                    max_lines=4,
                )
                gr.HTML('<div class="clinical-section-label"><span class="mi">🔊</span>Doctor voice response</div>')
                voice_out = gr.Audio(
                    label="Auto-plays for the patient",
                    autoplay=True,
                    elem_classes=["fit-media"],
                )
                timing_out = gr.Textbox(
                    label="Pipeline Timing (transcription | brain | TTS | total)",
                    lines=1,
                    elem_classes=["clinical-timing"],
                )
                with gr.Row():
                    download_btn = gr.DownloadButton("⬇ Download Summary", variant="secondary", size="sm")
                    forward_btn = gr.Button("➤ Forward to Doctor", variant="primary", size="sm")

        gr.HTML(FOOTER_HTML)

    # Wire pipeline: Analyze → 4 outputs
    analyze_btn.click(
        fn=process_inputs,
        inputs=[audio_in, image_in, video_in],
        outputs=[speech_out, guidance_out, voice_out, timing_out],
    )
    # Download summary from latest outputs
    download_btn.click(
        fn=_build_summary_file,
        inputs=[speech_out, guidance_out, timing_out],
        outputs=[download_btn],
    )
    # Clear everything
    clear_btn.click(
        fn=lambda: (None, None, None, "", "", None, ""),
        inputs=None,
        outputs=[audio_in, image_in, video_in, speech_out, guidance_out, voice_out, timing_out],
    )
    forward_btn.click(
        fn=lambda: gr.Info("Summary prepared — share the downloaded file with your doctor."),
        inputs=None,
        outputs=None,
    )

# Back-compat alias (old gr.Interface name)
iface = demo

if __name__ == "__main__":
    demo.launch(css=CLINICAL_CSS)
