import os
import threading
from typing import List, Dict, Optional
import soundfile as sf
import re

# We will load Kokoro centrally to avoid reloading the huge ONNX graph per file
_kokoro_instance = None
_kokoro_lock = threading.Lock()

def _get_kokoro():
    global _kokoro_instance
    if _kokoro_instance is None:
        with _kokoro_lock:
            if _kokoro_instance is None:
                from kokoro_onnx import Kokoro
                model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "kokoro-v1.0.onnx")
                voices_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "voices-v1.0.bin")
                
                if not os.path.exists(model_path) or os.path.getsize(model_path) < 300_000_000:
                    raise RuntimeError("Kokoro AI voice weights (310MB) are still downloading in the background. Please wait 1-2 minutes and try clicking Generate again!")
                if not os.path.exists(voices_path):
                    raise RuntimeError("Kokoro AI voices file is still downloading. Please wait!")
                    
                _kokoro_instance = Kokoro(model_path, voices_path)
    return _kokoro_instance

def list_voices() -> List[str]:
    return [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis"
    ]

def _apply_emotion_to_text(text: str, emotion: str) -> str:
    """
    Inject punctuation cues into text so Kokoro's model naturally
    produces more emotional breath patterns and rhythm.
    Kokoro is trained on real speech — it genuinely responds to these.
    """
    e = emotion.lower().strip()
    
    # Strip existing markdown and clean up first
    text = re.sub(r'[\*\_\~]+', '', text).strip()
    if not text:
        return text
    
    # Add trailing period if missing so sentence ends cleanly
    if text[-1] not in '.!?,;:':
        text += '.'

    if e in ("sad", "melancholic"):
        # Ellipses create natural trailing-off breath and hesitation
        text = re.sub(r'([,;])\s+', r'\1.. ', text)
        if text.endswith('.'):
            text = text[:-1] + '...'

    elif e == "angry":
        # Em-dashes create sharp clipped stops between thoughts
        text = re.sub(r'([,;])\s+', r' -- ', text)
        # Uppercase key exclamations
        text = re.sub(r'\b(never|stop|no|get out|how dare|enough)\b', 
                      lambda m: m.group().upper(), text, flags=re.IGNORECASE)

    elif e == "scared":
        # Short fragmented rhythm — dashes mimic stuttering/panic
        text = re.sub(r'([,;])\s+', r'-- ', text)
        text = re.sub(r'([.!?])\s+', r'\1 -- ', text)

    elif e in ("tense", "dramatic"):
        # Commas amplify the pauses already in text; add one before conjunctions
        text = re.sub(r'\s+(but|yet|still|however)\s+', r', \1 ', text, flags=re.IGNORECASE)

    elif e == "romantic":
        # Soft trailing ellipsis on longer phrases
        if len(text) > 40 and text.endswith('.'):
            text = text[:-1] + '...'

    elif e == "upbeat":
        # Upbeat: keep it punchy, strip soft trailing ellipsis if any
        text = text.replace('...', '.').replace('..', '.')

    return text.strip()

def _sanitize_for_kokoro(text):
    text = re.sub(r'[\*\_\~]+', '', text)
    return text.strip()

def synthesize_all(
    lines: List[Dict],
    voice_map: Dict[str, str],
    output_dir: str,
    progress_callback=None,
) -> List[Optional[str]]:
    """
    Generate audio for all lines sequentially using the local Kokoro ONNX model.
    Runs in a background thread to prevent blocking Streamlit.
    """
    os.makedirs(output_dir, exist_ok=True)
    default_voice = "af_heart"
    
    results: List[str] = []
    errors: List[Exception] = []
    
    # Pre-calculate to allocate paths
    paths = []
    valid_tasks = []
    
    for idx, line in enumerate(lines):
        character = line.get("character", "Narrator")
        text = line.get("text", "").strip()
        
        if not text:
            paths.append(None)
            continue
            
        voice = voice_map.get(character, default_voice)
        out_path = os.path.join(output_dir, f"line_{idx:05d}.wav")
        paths.append(out_path)
        valid_tasks.append((idx, text, voice, out_path, line.get("emotion", "neutral")))

    total_tasks = len(valid_tasks)
    completed_count = [0]

    def _run():
        try:
            kokoro = _get_kokoro()
            
            for i, (idx, text, voice, out_path, emotion) in enumerate(valid_tasks):
                # Cache skip
                if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
                    completed_count[0] += 1
                    continue
                
                # Exact speed mapping for the 10 emotion strings Gemini is instructed to return
                EMOTION_SPEED = {
                    "neutral":     0.95,
                    "happy":       1.10,
                    "sad":         0.78,
                    "tense":       1.15,
                    "dramatic":    1.05,
                    "romantic":    0.85,
                    "melancholic": 0.80,
                    "upbeat":      1.20,
                    "angry":       1.25,
                    "scared":      1.20,
                }
                speed = EMOTION_SPEED.get(emotion.lower().strip(), 0.95)
                    
                target_lang = "en-us"
                if voice.startswith("b"): target_lang = "en-gb"
                
                clean_text = _apply_emotion_to_text(text, emotion)
                if not clean_text:
                    continue
                    
                # Generate PCM Audio via ONNX
                try:
                    samples, sample_rate = kokoro.create(clean_text, voice=voice, speed=speed, lang=target_lang)
                    sf.write(out_path, samples, sample_rate)
                except Exception as ex:
                    print(f"Kokoro failed on chunk {idx}: {ex}")
                    
                completed_count[0] += 1

            results.extend(paths)
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    
    # UI thread safety polling layer
    while t.is_alive():
        t.join(timeout=0.2)
        if progress_callback and total_tasks > 0:
            progress_callback(completed_count[0], total_tasks)
            
    # Guarantee 100% update at the end
    if progress_callback and total_tasks > 0:
        progress_callback(total_tasks, total_tasks)
        
    if errors:
        raise errors[0]
        
    return paths
