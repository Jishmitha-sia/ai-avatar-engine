
# 🌱 Sprout: AI Avatar Engine

Sprout is a real-time, full-stack Conversational AI Avatar system. It combines Google's **Gemini 1.5** for intelligent conversation, **Edge-TTS** for voice synthesis, and **Wav2Lip** for highly accurate, real-time lip-syncing and video generation.

## ✨ Features
* **Ultra-Fast Generation:** Utilizes a "Warm Start" architecture by pre-loading AI models into GPU memory, dropping video generation latency to <5 seconds.
* **Multi-Avatar Support:** Dynamically switch between different tutor personas (images) and voice profiles in real-time.
* **Contextual Memory:** Maintains a continuous conversational thread using Gemini's ChatSession.
* **WhatsApp-Style Voice Input:** Speak directly to the avatar using the browser's Web Speech API.
* **"Life" Animations:** Avatars feature a subtle 3D breathing/scaling animation when idle to feel alive.

---

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed on your system:
1. **Python 3.8+**
2. **Node.js & npm** (For the frontend dashboard)
3. **FFmpeg** (CRITICAL for video processing)
	* **Windows:** `winget install ffmpeg`
	* **Mac:** `brew install ffmpeg`
	* **Linux:** `sudo apt install ffmpeg`

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sprout_avatar_project.git
cd sprout_avatar_project
```

### 2. Backend Setup (FastAPI & AI Engine)
Navigate to the backend directory and install the required Python packages:

```bash
cd backend
python -m venv venv
# On macOS / Linux:
source venv/bin/activate
# On Windows (PowerShell):
venv\Scripts\activate
pip install -r requirements.txt
```

⚠️ Important: Download the Wav2Lip Model Weights

The repository does not store the heavy machine learning weights.

Download the `wav2lip_gan.pth` file from the Official Wav2Lip Repository and place it here:

```
backend/Wav2Lip/checkpoints/wav2lip_gan.pth
```

Add your API Key:

Create a `.env` file inside the `backend/` folder and add your Google Gemini API key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Frontend Setup (React + Vite)
Open a new terminal window, navigate to the `frontend` directory, and install dependencies:

```bash
cd frontend
npm install
```

### 🏃‍♂️ Running the Application
You need to run both the Backend and Frontend servers simultaneously.

Terminal 1: Start the AI Backend

```bash
cd backend
# Make sure your virtual environment is activated
python main.py
```

(Wait until you see server logs indicating the backend is ready.)

Terminal 2: Start the Frontend UI

```bash
cd frontend
npm run dev
```

Open your browser and navigate to the local URL provided by Vite (usually http://localhost:5173).

## 📁 Project Structure
/backend: FastAPI server, Gemini logic, and Wav2Lip integration.

/avatars: Drop any .jpg or .png here to instantly add new avatars.

/Wav2Lip: The modified lip-sync AI engine (`sprout_engine.py`).

/frontend: React application using Tailwind CSS and Lucide icons.

## 📝 Troubleshooting

- 404 Video Not Found / Video isn't generating: Ensure FFmpeg is installed and added to your system's PATH.
- Engine Start Failed: Ensure `wav2lip_gan.pth` is placed in the exact checkpoints folder mentioned above.
- Microphone not working: Ensure you are accessing the frontend via localhost or HTTPS, as browsers block microphone access on insecure HTTP connections.

