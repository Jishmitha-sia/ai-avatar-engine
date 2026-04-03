# 🌱 Sprout: The AI Avatar Tutor (v2.0)

Sprout is a high-performance, full-stack **Conversational AI Avatar Dashboard**. It transforms any PDF or text query into a live, interactive educational experience using a 136x faster video inference engine.

---

## 🔥 What's New in Sprout 2.0?

Sprout 2.0 is a massive upgrade focused on deep knowledge and interactivity:

*   **🧠 Knowledge Deep-Dive (10 Points):** A 2-column grid dashboard that extracts up to 10 key concepts from any document or chat in real-time.
*   **🎯 AI Quiz Mode:** An interactive examiner! Sprout generates a bespoke 3-question multiple-choice quiz based on the context of your conversation.
*   **📂 PDF-to-Video Pipeline:** Upload any PDF to get an instant, lip-synced video summary and a structured "Live Knowledge Board".
*   **📝 Study Guide Export:** One-click download of all extracted concepts as a formatted study guide.
*   **🖥️ Command Center UI:** A dynamic widescreen (1500px) layout with a "Maximize" toggle for a truly immersive experience.
*   **🚀 Blackwell-Ready Engine:** Optimized specifically for NVIDIA's latest **RTX 50-series (Blackwell)** GPUs for ultra-low latency.

---

## ✨ Core Features

*   **Ultra-Fast Generation:** Direct FFmpeg piping drops video latency to **< 5 seconds** for 20s clips.
*   **Accurate Lip-Sync:** Powered by a customized Wav2Lip implementation that handles odd-numbered image dimensions and high-res avatars.
*   **Natural Voice:** Integration with `edge-tts` for high-fidelity, human-like narration.
*   **Subtle Animations:** Avatars feature "Breathing" and "Idle" animations to feel alive while waiting for your input.

---

## ⚙️ Prerequisites

1.  **Python 3.11+** (Recommended for Blackwell/sm_120 compatibility)
2.  **Node.js & npm** (For the React Dashboard)
3.  **FFmpeg** (Ensure it's in your system's PATH)
    *   `winget install ffmpeg` (Windows)
4.  **Hardware:** NVIDIA GPU (RTX 30/40/50 series) with 8GB+ VRAM recommended.

---

## 🚀 Installation & Setup

### 1. Backend Setup
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Weights & API Keys:**
*   **Wav2Lip Weights:** Place `wav2lip_gan.pth` in `backend/Wav2Lip/checkpoints/`.
*   **S3FD Face Detector:** Place `s3fd.pth` in `backend/Wav2Lip/face_detection/detection/sfd/`.
*   **API Key:** Create a `.env` file in the `backend/` folder:
    ```env
    GEMINI_API_KEY=your_key_here
    ```

### 2. Frontend Setup
```bash
cd frontend
npm install
```

---

## 🏃‍♂️ Running the Dashboard

### Terminal 1: The Engine
```bash
cd backend
.\venv\Scripts\activate
python main.py
```
*(Watch the console for the "MEMORY & STATIC FILES READY" message.)*

### Terminal 2: The Interface
```bash
cd frontend
npm run dev
```

---

## 📂 Project Structure
*   `backend/main.py`: The FastAPI server & Gemini logic.
*   `backend/Wav2Lip/sprout_engine.py`: The high-speed video generation core.
*   `backend/avatars/`: The Persona library. Add any `.jpg` here!
*   `frontend/src/App.jsx`: The "Command Center" React UI.

---

## 📝 Developer Notes & Best Practices

*   **Avoid Odd Dimensions:** Sprout now handles odd heights automatically, but for best performance, use even-numbered resolutions (e.g., 512x512).
*   **GPU Warm-up:** The first generation on a new PC may take longer (30s) as Torch JIT-compiles kernels for your specific GPU architecture. Subsequent runs will be sub-5 seconds.
*   **Microphone Access:** Ensure you grant browser permissions. Sprout works best in Chrome/Edge.

---

## 📜 Credits & License
- **AI Core:** Google Gemini 1.5.
- **Lip-Sync Research:** Inspired by Rudrabha/Wav2Lip.
- **Voice Synthesis:** Microsoft Edge-TTS.

*Built with ❤️ for High-Performance Educational AI.*
