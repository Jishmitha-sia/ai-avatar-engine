import os
import uvicorn
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import edge_tts
import asyncio

# 1. Setup Environment
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 2. Dynamic Model Selection Logic
def get_working_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                return m.name
    except: pass
    return "models/gemini-1.5-flash-latest"

WORKING_MODEL_NAME = get_working_model()

# 3. The Video Generator Function (Phase 2 R&D)
def generate_video():
    print("🎬 RTX GPU: Starting Lip-Sync Generation...")
    # This command triggers the Wav2Lip engine locally on your GPU
    command = [
        "python", "Wav2Lip/inference.py",
        "--checkpoint_path", "Wav2Lip/checkpoints/wav2lip_gan.pth",
        "--face", "avatar.jpg",
        "--audio", "response.mp3",
        "--outfile", "output_video.mp4"
    ]
    # Check if files exist before running to avoid errors
    if not os.path.exists("avatar.jpg") or not os.path.exists("Wav2Lip/checkpoints/wav2lip_gan.pth"):
        print("❌ Error: Missing avatar.jpg or wav2lip_gan.pth weights.")
        return

    subprocess.run(command, check=True)
    print("✅ Video generated successfully.")

@app.post("/chat")
async def chat_with_sprout(user_query: str):
    print(f"\n--- Sprout Request: {user_query} ---")
    try:
        # A. Intelligence Layer (Gemini)
        model = genai.GenerativeModel(WORKING_MODEL_NAME)
        response = model.generate_content(f"You are Sprout, a professional adult tutor. Answer in 2 short sentences: {user_query}")
        ai_text = response.text
        print(f"AI Text: {ai_text}")
        
        # B. Voice Layer (Professional Adult Jenny Voice)
        communicate = edge_tts.Communicate(ai_text, "en-US-JennyNeural")
        await communicate.save("response.mp3")
        
        # C. Visual Layer (Trigger the GPU subprocess)
        # We use run_in_executor to keep the API responsive while the GPU works
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
    if os.path.exists("output_video.mp4"):
        return FileResponse("output_video.mp4", media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not ready")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)