import os, threading
from src.tts.kokoro_tts_generator import _get_kokoro, list_voices
import soundfile as sf

def main():
    kokoro = _get_kokoro()
    voices = list_voices()
    
    os.makedirs("output/samples", exist_ok=True)
    
    for v in voices:
        out = f"output/samples/{v}.wav"
        if not os.path.exists(out):
            print(f"Generating {v}...")
            lang = "en-gb" if v.startswith("b") else "en-us"
            try:
                samples, sr = kokoro.create("Hello, I am testing this voice.", voice=v, speed=1.0, lang=lang)
                sf.write(out, samples, sr)
            except Exception as e:
                print(e)
                
main()
