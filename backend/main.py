import os
import sys
import uvicorn
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import edge_tts
import asyncio
import shutil

# 1. Setup Environment
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

def get_working_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                return m.name
    except Exception: pass
    return "models/gemini-1.5-flash-latest"

WORKING_MODEL_NAME = get_working_model()

# 2. Robust Video Generation
def generate_video():
    print("🎬 Starting Lip-Sync Generation...")
    
    # Check for FFmpeg first
    if not shutil.which("ffmpeg"):
        print("❌ CRITICAL: FFmpeg not found! Please install it with: winget install Gyan.FFmpeg")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    wav2lip_dir = os.path.join(base_dir, "Wav2Lip")
    
    face_path = os.path.join(base_dir, "avatar.jpg")
    audio_path = os.path.join(base_dir, "response.mp3")
    checkpoint_path = os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth")
    output_path = os.path.join(base_dir, "output_video.mp4")

    # Verify model size again
    size = os.path.getsize(checkpoint_path)
    print(f"📦 Model File Size: {size / (1024*1024):.2f} MB")

    command = [
        sys.executable, "inference.py",
        "--checkpoint_path", checkpoint_path,
        "--face", face_path,
        "--audio", audio_path,
        "--outfile", output_path
    ]

    try:
        print(f"⏳ Processing frames (Streaming Logs Below)...")
        # Use shell=False for safety; run in Wav2Lip directory
        process = subprocess.run(command, cwd=wav2lip_dir, check=True)
        
        if os.path.exists(output_path):
            print(f"✅ Video truly generated at: {output_path}")
        else:
            print(f"❌ Video file missing! Wav2Lip finished but the file isn't at {output_path}")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Wav2Lip process failed with exit code {e.returncode}")

@app.post("/chat")
async def chat_with_sprout(user_query: str):
    print(f"\n--- Request: {user_query} ---")
    try:
        model = genai.GenerativeModel(WORKING_MODEL_NAME)
        response = model.generate_content(f"You are a professional tutor. Answer in 2 short sentences: {user_query}")
        ai_text = response.text
        print(f"AI Text: {ai_text}")
        
        audio_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "response.mp3")
        communicate = edge_tts.Communicate(ai_text, "en-US-JennyNeural")
        await communicate.save(audio_save_path)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, generate_video)
        
        return {
            "text": ai_text,
            "video_url": "http://127.0.0.1:8000/get-video"
        }
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-video")
def get_video():
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_video.mp4")
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found on server.")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)