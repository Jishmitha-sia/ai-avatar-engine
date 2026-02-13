import os, sys, uvicorn, shutil, asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import edge_tts

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.append(os.path.join(base_dir, "Wav2Lip"))
from Wav2Lip.sprout_engine import SproutEngine

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sprout_engine = None

# --- CONFIGURATION ---
VOICES = [
    {"id": "en-US-JennyNeural", "name": "Jenny (Female US)", "gender": "Female"},
    {"id": "en-US-GuyNeural", "name": "Guy (Male US)", "gender": "Male"},
    {"id": "en-GB-SoniaNeural", "name": "Sonia (Female UK)", "gender": "Female"},
    {"id": "en-US-ChristopherNeural", "name": "Christopher (Male US)", "gender": "Male"}
]

@app.on_event("startup")
async def startup_event():
    global sprout_engine
    print("🌱 STARTING SPROUT MULTI-AVATAR SERVER...")
    
    checkpoint = f"{base_dir}/Wav2Lip/checkpoints/wav2lip_gan.pth"
    avatars_dir = f"{base_dir}/avatars"
    
    if not os.path.exists(avatars_dir):
        os.makedirs(avatars_dir)
        print(f"⚠️ Created missing avatars folder at: {avatars_dir}")

    try:
        sprout_engine = SproutEngine(checkpoint, avatars_dir)
        print("✅ SERVER READY")
    except Exception as e:
        print(f"❌ Engine Start Failed: {e}")

@app.get("/config")
def get_config():
    """Returns available avatars and voices to the frontend"""
    if not sprout_engine:
        return {"avatars": [], "voices": VOICES}
    
    # List all cached filenames
    available_avatars = list(sprout_engine.avatar_cache.keys())
    return {
        "avatars": available_avatars,
        "voices": VOICES
    }

@app.post("/chat")
async def chat(user_query: str, avatar_id: str = None, voice_id: str = "en-US-JennyNeural"):
    print(f"\n--- Chat: {user_query} | Avatar: {avatar_id} | Voice: {voice_id} ---")
    
    if not sprout_engine:
        raise HTTPException(500, "Engine not active")

    # Default to first available avatar if none selected
    if not avatar_id:
        if not sprout_engine.avatar_cache:
            raise HTTPException(500, "No avatars loaded on server")
        avatar_id = list(sprout_engine.avatar_cache.keys())[0]

    try:
        # 1. Gemini (FIXED MODEL NAME HERE)
        # We use the standard 'gemini-1.5-flash' which is stable and fast
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"You are a helpful tutor. Answer briefly: {user_query}")
        ai_text = response.text
        
        # 2. TTS
        audio_path = f"{base_dir}/response.mp3"
        communicate = edge_tts.Communicate(ai_text, voice_id)
        await communicate.save(audio_path)
        
        # 3. Video
        output_path = f"{base_dir}/output_video.mp4"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, sprout_engine.infer, audio_path, output_path, avatar_id)
        
        return {"text": ai_text, "video_url": "http://127.0.0.1:8000/get-video"}
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(500, str(e))

@app.get("/get-video")
def get_video():
    path = f"{base_dir}/output_video.mp4"
    if os.path.exists(path): return FileResponse(path, media_type="video/mp4")
    raise HTTPException(404, "Video not found")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)