# 🌱 Sprout-tutor

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-20%2B-green.svg)](https://nodejs.org/)

**Sprout** is a high‑performance, full‑stack AI Avatar Dashboard that turns PDFs or text queries into live, interactive educational experiences. It combines Gemini LLM, Edge‑TTS, and ultra‑fast video generation (SadTalker / Wav2Lip) to deliver instant, lip‑synced avatar videos.

---

## Features

- **Multi‑language chat** – Supports English, Spanish, French, Hindi, Chinese, German with concise responses.
- **PDF‑to‑Video pipeline** – Upload a PDF and receive a summarized script, key concepts, and a generated avatar video.
- **Voice‑Chat** – Send an audio message, get transcription, answer, concepts, and avatar video.
- **Quiz Generation** – Auto‑creates a 3‑question multiple‑choice quiz from the conversation.
- **Avatar serving** – Static avatar images are exposed via `/avatars`.
- **Dynamic pricing constants** – Cost per token for input/output tracking.
- **Fast video generation** – Uses SadTalker (or legacy Wav2Lip) with GPU acceleration (< 5 s latency for 20 s clips).
- **CORS enabled** – Frontend can call the backend without restrictions.

---

## Architecture

```mermaid
flowchart LR
    subgraph Frontend[React Frontend]
        FE[App.jsx] -->|REST API| BE[FastAPI Backend]
    end
    subgraph Backend[FastAPI Backend]
        BE -->|Gemini LLM| Gemini[google.genai]
        BE -->|Edge‑TTS| TTS[edge_tts]
        BE -->|FFmpeg| FFmpeg[ffmpeg]
        BE -->|SadTalker| SadTalker[SadTalker Engine]
        BE -->|Wav2Lip| Wav2Lip[Wav2Lip Engine]
        BE -->|Static| Avatars[/avatars]
    end
    classDef cloud fill:#f9f,stroke:#333,stroke-width:2px;
    class Gemini,TTS,FFmpeg,SadTalker,Wav2Lip cloud;
```

---

## Prerequisites

- **Python 3.11+**
- **Node.js 20+** & **npm**
- **FFmpeg** (must be in `PATH` – `winget install ffmpeg` on Windows)
- **NVIDIA GPU** (RTX 30/40/50 series, 8 GB+ VRAM recommended)

---

## Installation & Setup

### Backend
```bash
cd backend
python -m venv venv
.\\venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```
Create a `.env` file (or use the root one) with your Gemini API key:
```env
GEMINI_API_KEY=your_key_here
```
Place required model weights:
- `backend/Wav2Lip/checkpoints/wav2lip_gan.pth`
- `backend/SadTalker/checkpoints/` (download from the SadTalker repo)

### Frontend
```bash
cd frontend
npm install
npm run dev   # starts Vite dev server on http://localhost:5173
```
The frontend will proxy API calls to `http://127.0.0.1:8000`.

---

## Quick Start (run both services)
```bash
# Terminal 1 – Backend
cd backend
.\\venv\\Scripts\\activate && uvicorn main:app --reload

# Terminal 2 – Frontend
cd frontend
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## API Usage Examples
### Get configuration (avatars, voices, languages)
```bash
curl http://127.0.0.1:8000/config
```
### Chat (text)
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_query": "Explain photosynthesis", "language": "en"}'
```
### PDF‑to‑Video
```bash
curl -X POST "http://127.0.0.1:8000/pdf-to-video" \
  -F "file=@/path/to/document.pdf" \
  -F "language=en"
```
### Voice‑Chat (audio file)
```bash
curl -X POST "http://127.0.0.1:8000/voice-chat" \
  -F "file=@/path/to/audio.webm" \
  -F "language=en"
```
### Generate Quiz
```bash
curl "http://127.0.0.1:8000/generate-quiz?language=en"
```
All responses include `text`, `concepts`, and a `video_url` pointing to `/get-video`.

---

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-feature`).
3. Follow the existing code style (black, isort, eslint).
4. Submit a Pull Request with a clear description.

---

## License

MIT License – see the [LICENSE](LICENSE) file.

---

## Credits
- **AI Core:** Google Gemini 1.5
- **Lip‑Sync:** Rudrabha/Wav2Lip
- **Voice Synthesis:** Microsoft Edge‑TTS
- **Video Engine:** SadTalker (Lively avatar animation)


