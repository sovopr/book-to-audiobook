"""
AI Audiobook Maker
Streamlit application — main entry point.
"""
import json
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.epub_parser import parse_epub
from src.ingestion.text_cleaner import clean_text
from src.nlp.chunker import chunk_text
from src.nlp.dialogue_extractor import extract_dialogue, test_connection
from src.tts.kokoro_tts_generator import list_voices, synthesize_all
from src.mixing.audio_mixer import mix_audiobook

load_dotenv()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Audiobook Maker",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS — Premium dark theme
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b27 50%, #0d1117 100%);
        min-height: 100vh;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b27 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }

    /* Hero title */
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        margin-bottom: 0.25rem;
    }

    .hero-sub {
        color: #8b949e;
        font-size: 1.05rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }

    /* Cards */
    .card {
        background: linear-gradient(145deg, #1c2333, #161b27);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s ease;
    }
    .card:hover { border-color: #6e7681; }

    .card-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 0.75rem;
    }

    /* Step badges */
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #7c3aed, #3b82f6);
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.6rem;
    }

    .step-label {
        font-size: 1rem;
        font-weight: 600;
        color: #e6edf3;
    }

    /* Status pills */
    .pill-done   { background:#1a3a2a; color:#56d364; border:1px solid #2ea043; border-radius:999px; padding:2px 12px; font-size:0.8rem; }
    .pill-active { background:#1c2c4d; color:#79c0ff; border:1px solid #388bfd; border-radius:999px; padding:2px 12px; font-size:0.8rem; }
    .pill-wait   { background:#21262d; color:#8b949e; border:1px solid #30363d; border-radius:999px; padding:2px 12px; font-size:0.8rem; }

    /* Button overrides */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.6rem !important;
        transition: opacity 0.2s ease !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #161b27 !important;
        border: 2px dashed #30363d !important;
        border-radius: 12px !important;
    }

    /* Inputs */
    input, textarea, select {
        background-color: #161b27 !important;
        border-color: #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
    }

    /* Progress bar */
    .stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #3b82f6) !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────
defaults = {
    "cleaned_text": None,
    "dialogue_lines": None,
    "audio_paths": None,
    "voice_map": {},
    "available_voices": [],
    "characters": [],
    "output_path": None,
    "step": 1,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# Sidebar — config
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="card-title">⚙️ Configuration</p>', unsafe_allow_html=True)

    gemini_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", "REDACTED"),
        type="password",
        placeholder="AIza...",
        help="Get a free key at aistudio.google.com",
    )

    gemini_model = st.selectbox(
        "Gemini Model",
        [
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-image-preview",
        ],
        index=0,
        help="gemini-3-flash-preview is recommended — fast and free.",
    )

    st.markdown("---")
    st.markdown('<p class="card-title">🎵 Audio Settings</p>', unsafe_allow_html=True)
    enable_music = st.toggle("Background Music", value=False)
    st.caption("Royalty-free ambient tracks auto-matched to scene emotion.")

    st.markdown("---")
    st.markdown(
        '<p style="color:#8b949e;font-size:0.78rem;">TTS powered by <b>Kokoro-82M ONNX</b><br>'
        'Free · High Fidelity · Local Inference</p>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Main — Hero
# ─────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🎙️ AI Audiobook Maker</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Transform any PDF or EPUB into a fully voiced, emotionally scored audiobook — completely free.</p>',
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([3, 2], gap="large")

# ─────────────────────────────────────────────
# STEP 1 — Upload & Extract
# ─────────────────────────────────────────────
with col_left:
    with st.container():
        st.markdown(
            '<div class="card">'
            '<p class="card-title"><span class="step-badge">1</span> Upload Book</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        project_name = st.text_input(
            "Project Name",
            placeholder="e.g. Pride and Prejudice",
            label_visibility="collapsed",
        )

        uploaded = st.file_uploader(
            "Drag & drop a PDF or EPUB",
            type=["pdf", "epub"],
            label_visibility="collapsed",
        )

        if uploaded and st.button("📄 Extract & Clean Text", use_container_width=True):
            with st.spinner("Parsing document…"):
                raw_bytes = uploaded.read()
                if uploaded.name.lower().endswith(".pdf"):
                    raw_text = parse_pdf(raw_bytes)
                else:
                    raw_text = parse_epub(raw_bytes)
                st.session_state.cleaned_text = clean_text(raw_text)
                st.session_state.step = 2
            st.rerun()  # force re-render so Step 2 appears immediately

        if st.session_state.cleaned_text:
            with st.expander("📖 Preview extracted text"):
                st.text_area(
                    "Cleaned text",
                    st.session_state.cleaned_text[:3000] + ("…" if len(st.session_state.cleaned_text) > 3000 else ""),
                    height=200,
                    label_visibility="collapsed",
                )

# ─────────────────────────────────────────────
# STEP 2 — NLP Dialogue Extraction
# ─────────────────────────────────────────────
    if st.session_state.cleaned_text:
        st.markdown(
            '<div class="card" style="margin-top:1rem;">'
            '<p class="card-title"><span class="step-badge">2</span> NLP Speaker Attribution</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        if not gemini_key:
            st.warning("⚠️ Enter your Gemini API key in the sidebar to run NLP extraction.")
        else:
            if st.button("🧠 Extract Dialogue & Speakers", use_container_width=True):
                # Quick API connectivity check first
                with st.spinner("Testing Gemini connection…"):
                    try:
                        test_out = test_connection(gemini_key, gemini_model)
                        if not test_out or "error" in test_out.lower()[:30]:
                            st.error(f"❌ Gemini connection test returned unexpected output: {test_out[:300]}")
                            st.stop()
                    except Exception as te:
                        st.error(f"❌ Cannot reach Gemini API: {te}")
                        st.stop()

                chunks = chunk_text(st.session_state.cleaned_text)
                # Progress bar OUTSIDE spinner so it can update live
                prog = st.progress(0, text=f"Processing chunk 0/{len(chunks)}…")
                status_text = st.empty()

                def nlp_progress(cur, tot):
                    prog.progress(cur / tot, text=f"Processing chunk {cur}/{tot}…")

                status_text.caption(f"Sending {len(chunks)} chunks to Gemini…")
                lines, extraction_errors = extract_dialogue(
                    chunks,
                    api_key=gemini_key,
                    model_name=gemini_model,
                    progress_callback=nlp_progress,
                )
                prog.progress(1.0, text="✅ Extraction complete!")
                status_text.empty()

                if extraction_errors:
                    with st.expander(f"⚠️ {len(extraction_errors)} chunk(s) had errors (click to view)"):
                        for err in extraction_errors:
                            st.code(err)

                if not lines:
                    st.error(
                        "❌ Extraction returned 0 lines. This usually means the model output wasn't "
                        "valid JSON. Check the errors above, or try a different Gemini model in the sidebar."
                    )
                else:
                    st.session_state.dialogue_lines = lines
                    st.session_state.characters = sorted(
                        {l["character"] for l in lines if l.get("character")}
                    )
                    st.session_state.step = 3
                    st.rerun()  # force re-render so col_right (Step 3) appears

        if st.session_state.dialogue_lines is not None:
            st.success(f"✅ Extracted {len(st.session_state.dialogue_lines)} lines from {len(st.session_state.characters)} characters")
            with st.expander(f"💬 View extracted lines"):
                st.json(st.session_state.dialogue_lines[:20])

# ─────────────────────────────────────────────
# STEP 3 — Voice Mapping
# ─────────────────────────────────────────────
with col_right:
    if st.session_state.dialogue_lines:
        st.markdown(
            '<div class="card">'
            '<p class="card-title"><span class="step-badge">3</span> Voice Mapping</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Load voices once
        if not st.session_state.available_voices:
            with st.spinner("Loading neural voices…"):
                try:
                    st.session_state.available_voices = list_voices()
                except Exception as e:
                    st.error(f"Could not load voices: {e}")
                    st.session_state.available_voices = ["af_heart", "am_adam", "bf_emma"]

        voices = st.session_state.available_voices
        default_voices = {
            "Narrator": "am_adam",
        }

        # Auto-assignment state
        if "auto_voice_map" not in st.session_state:
            st.session_state.auto_voice_map = {}

        if st.button("🤖 Auto-Assign Voices with AI", use_container_width=True):
            if not st.session_state.characters or not voices:
                st.warning("No characters or voices available.")
            else:
                if len(st.session_state.characters) > 20:
                    st.info(f"⏳ Mapping {len(st.session_state.characters)} characters. This may take 30–60 seconds. Please do not refresh...")
                    
                with st.spinner("Analyzing characters and mapping ideal neural voices..."):
                    from src.nlp.voice_assigner import auto_assign_voices
                    try:
                        new_map = auto_assign_voices(
                            st.session_state.characters,
                            voices,
                            api_key=gemini_key,
                            model_name=gemini_model,
                        )
                        st.session_state.auto_voice_map = new_map
                        # Force Streamlit to overwrite existing widget selections from the previous run
                        for char, assigned_voice in new_map.items():
                            st.session_state[f"voice_{char}"] = assigned_voice
                            
                        st.success("Successfully auto-mapped voices!")
                        # Force a rerun to lock in the new widget states visually
                        st.rerun()
                    except Exception as e:
                        st.error(f"Auto-assign failed: {e}")

        import collections

        st.caption("Assign a neural voice to each detected character:")
        voice_map = {}
        for char in st.session_state.characters:
            st.markdown(f"**🎭 {char}**")
            col1, col2 = st.columns([2, 3])
            
            # Prepare kwargs to avoid Streamlit's Session State API warning.
            # We only supply 'index' if the widget hasn't been instantiated/saved in session_state.
            widget_key = f"voice_{char}"
            kwargs = {
                "options": voices,
                "key": widget_key,
                "label_visibility": "collapsed"
            }
            if widget_key not in st.session_state:
                default = st.session_state.auto_voice_map.get(char)
                if not default:
                    default = default_voices.get(char, voices[0] if voices else "af_heart")
                default_idx = voices.index(default) if default in voices else 0
                kwargs["index"] = default_idx

            with col1:
                voice_map[char] = st.selectbox(
                    "Select Voice",
                    **kwargs
                )
            with col2:
                sample_path = os.path.join(os.path.dirname(__file__), "output", "samples", f"{voice_map[char]}.wav")
                if os.path.exists(sample_path):
                    st.audio(sample_path)
            st.markdown("---")

        # Duplicate character voice checking
        voice_counts = collections.Counter(voice_map.values())
        duplicates = [v for v, c in voice_counts.items() if c > 1]
        
        if duplicates:
            warning_msg = "⚠️ **Duplicate Voices Detected:**\n\n"
            for dup_voice in duplicates:
                chars_with_voice = [char for char, voice in voice_map.items() if voice == dup_voice]
                warning_msg += f"- **{dup_voice}** is assigned to: {', '.join(chars_with_voice)}\n"
            warning_msg += "\n*For the most natural audiobook experience, try assigning a unique voice to every character!*"
            st.warning(warning_msg)
        st.session_state.voice_map = voice_map

        # ─────────────────────────────────────────
        # STEP 4 — Generate Audiobook
        # ─────────────────────────────────────────
        st.markdown(
            '<div class="card" style="margin-top:1rem;">'
            '<p class="card-title"><span class="step-badge">4</span> Generate Audiobook</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.button("🚀 Generate Audiobook", use_container_width=True):
            if not project_name:
                st.warning("Please enter a project name first.")
            else:
                out_dir = os.path.join(
                    os.path.dirname(__file__), "output", "tmp", project_name.replace(" ", "_")
                )
                os.makedirs(out_dir, exist_ok=True)

                # --- TTS ---
                tts_prog = st.progress(0, text="🎙️ Synthesizing voices…")
                st.caption("🎙️ Generating audio clips natively via Kokoro-82M ONNX Engine…")

                def tts_progress(cur, tot):
                    tts_prog.progress(cur / tot, text=f"Synthesizing line {cur}/{tot}…")

                audio_paths = synthesize_all(
                    st.session_state.dialogue_lines,
                    st.session_state.voice_map,
                    out_dir,
                    progress_callback=tts_progress,
                )
                tts_prog.progress(1.0, text="✅ Voice synthesis complete!")
                st.session_state.audio_paths = audio_paths

                # --- Mix ---
                mix_prog = st.progress(0, text="🎵 Mixing audio…")

                final_output = os.path.join(
                    os.path.dirname(__file__), "output",
                    f"{project_name.replace(' ', '_').replace("'", '')}.mp3"
                )

                def mix_progress(cur, tot):
                    mix_prog.progress(cur / tot, text=f"Mixing line {cur}/{tot}…")

                # Temporarily disable music overlay if toggle is off
                if not enable_music:
                    # Monkey-patch emotion_mapper to return None (no music)
                    import src.sound.emotion_mapper as em
                    _orig = em.get_music_path
                    em.get_music_path = lambda e: None

                mix_audiobook(
                    st.session_state.dialogue_lines,
                    audio_paths,
                    final_output,
                    progress_callback=mix_progress,
                    enable_music=enable_music,
                )

                if not enable_music:
                    em.get_music_path = _orig

                mix_prog.progress(1.0, text="✅ Mixing complete!")
                st.session_state.output_path = final_output
                st.session_state.step = 5

        # ─────────────────────────────────────────
        # STEP 5 — Download
        # ─────────────────────────────────────────
        if st.session_state.output_path and os.path.exists(st.session_state.output_path):
            st.markdown("---")
            st.success("🎉 Your audiobook is ready!")
            fname = os.path.basename(st.session_state.output_path)
            with open(st.session_state.output_path, "rb") as f:
                st.download_button(
                    label=f"⬇️ Download {fname}",
                    data=f,
                    file_name=fname,
                    mime="audio/mpeg",
                    use_container_width=True,
                )
            size_mb = os.path.getsize(st.session_state.output_path) / 1_000_000
            st.caption(f"📦 File size: {size_mb:.1f} MB")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#8b949e;font-size:0.8rem;">'
    'Powered by Gemini · Kokoro-onnx · pydub &nbsp;|&nbsp; 100% Free TTS · No API key for voices'
    '</p>',
    unsafe_allow_html=True,
)
