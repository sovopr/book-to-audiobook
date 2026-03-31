# book-to-audiobook

Converts a PDF book into an audiobook using local AI voice synthesis.

Extracts dialogue from the book using Gemini, assigns a voice to each character, and synthesizes everything using the Kokoro TTS model running locally via ONNX. No cloud TTS, no API costs for audio.

## How it works

1. Upload a PDF
2. Gemini chunks and extracts dialogue with character names and emotion tags
3. You pick a voice for each character (there are previews so you can hear them first)
4. Kokoro generates the audio clips locally
5. ffmpeg glues everything into one MP3

Previous runs are cached so if you stop halfway you don't have to start over.

## Setup

You need `ffmpeg` installed:
```bash
brew install ffmpeg
```

Install Python deps:
```bash
pip install -r requirements.txt
pip install kokoro-onnx soundfile
```

Download the Kokoro model weights (about 310MB):
```bash
mkdir -p models && cd models
curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd ..
```

Add your Gemini API key to a `.env` file:
```
GEMINI_API_KEY=your_key
```

## Running

```bash
streamlit run app.py
```

If you want it to keep running after you close the terminal:
```bash
nohup streamlit run app.py --server.headless true < /dev/null > nohup.out 2>&1 &
```

## Voices

Uses Kokoro v1.0 voices — American (`af_heart`, `am_adam`...) and British (`bf_emma`, `bm_daniel`...). You can preview each one in the UI before assigning.

## Notes

- The model runs on your CPU/Apple Neural Engine, not sent to any cloud
- Generation speed is roughly 5-10x real-time on M-series chips
- Background music is off by default
