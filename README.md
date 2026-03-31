# 🎙️ AI Audiobook Maker (Kokoro-82M Native)

A lightning-fast, production-ready AI audiobook generator built strictly for Apple Silicon (M-series) Macs. This pipeline completely eliminates cloud dependency by running the State-of-the-Art **Kokoro-82M v1.0** neural voice model locally via ONNX Runtime. 

It takes any PDF novel, uses Google Gemini to semantically extract thousands of lines of dialogue with emotional tags, synthesizes hyper-realistic human voice audio (with emotional speed modulation), and binds them together using a multi-threaded `FFMpeg` fast-concatenate engine in seconds.

## ✨ Features
* **100% Offline Synthesis:** Uses the blazing-fast Kokoro v1.0 ONNX model directly on your CPU/Neural Engine.
* **Smart Voice Casting:** Automatically parses characters from the narrative and binds them to distinct neural voice actors (`af_heart`, `am_adam`, etc.).
* **SSML Emotional Prosody:** Dynamically slows down or speeds up the voice acting based on Gemini's emotional detection (angry, sad, terrified) for incredibly human-like pacing.
* **Instant Assembly:** Bypasses slow legacy O(N^2) memory reallocation by directly piping raw `.wav` neural output into a fast `ffmpeg -c:a libmp3lame` compilation engine.
* **State Caching:** Interrupt the pipeline at any time; progress is automatically cached to disk, enabling 0-second re-synthesizing!

## 🚀 Installation 

### 1. Requirements
* macOS (M1/M2/M3 recommended for ONNX speed)
* Python 3.10+
* `ffmpeg` installed globally (`brew install ffmpeg`)

### 2. Setup
Clone the repository and install the Python dependencies:
```bash
git clone https://github.com/sovopr/book-to-audiobook.git
cd book-to-audiobook

pip install -r requirements.txt
pip install kokoro-onnx soundfile
```

### 3. Download the AI Weights
Kokoro requires the 310MB ONNX neural graph and voice registry. Run this in your terminal to download them into the `models/` folder:
```bash
mkdir -p models && cd models
curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd ..
```

### 4. API Key
Create a `.env` file in the root directory and add your Google Gemini API key (used exclusively for text parsing, not audio generation):
```env
GEMINI_API_KEY=your_key_here
```

## 🎧 Usage
Launch the Streamlit web interface locally:
```bash
streamlit run app.py
```
> **Pro Tip:** If you want to run the server in the background and close your terminal without macOS freezing the `ffmpeg` subprocesses (via `SIGTTIN`), launch it detached:
> `nohup streamlit run app.py --server.headless true < /dev/null > nohup.out 2>&1 &`

1. Upload your PDF novel.
2. The AI will chunk the text, extract dialogue, and detect emotions.
3. Preview and assign Kokoro voices to each character (or click Auto-Assign).
4. Uncheck Background Music (optional).
5. Click **Generate Audiobook**. The pipeline will synthesize the lines at roughly 5x real-time speed and mix your multi-hour MP3!

## 🛠 Architecture Stack
- **Web UI:** Streamlit
- **Text Extraction:** `pymupdf` (FitZ)
- **NLP / Emotion Tagging:** `google-genai` (Gemini 2.5 Flash)
- **Local Audio Synthesis:** `kokoro-onnx` (82 million param) + `soundfile`
- **Audio Mixing Engine:** `ffmpeg` Direct Disk-Concat Demuxer
