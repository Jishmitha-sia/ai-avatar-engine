import os, sys, uvicorn, shutil, asyncio, time, fitz
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles # <--- NEW IMPORT
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import edge_tts

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.append(os.path.join(base_dir, "Wav2Lip"))
from Wav2Lip.sprout_engine import SproutEngine
import json, re

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- NEW: SERVE AVATAR IMAGES ---
avatars_dir = f"{base_dir}/avatars"
if not os.path.exists(avatars_dir): os.makedirs(avatars_dir)
app.mount("/avatars", StaticFiles(directory=avatars_dir), name="avatars") # <--- NEW MOUNT

# Global Variables
sprout_engine = None
chat_session = None

# --- PRICING CONSTANTS ---
COST_INPUT_1M = 0.075
COST_OUTPUT_1M = 0.30

# --- CONFIGURATION ---
VOICES = [
    {"id": "en-US-JennyNeural", "name": "Jenny (Female US)", "gender": "Female"},
    {"id": "en-US-GuyNeural", "name": "Guy (Male US)", "gender": "Male"},
    {"id": "en-GB-SoniaNeural", "name": "Sonia (Female UK)", "gender": "Female"},
    {"id": "en-US-ChristopherNeural", "name": "Christopher (Deep/Calm)", "gender": "Male"},
    {"id": "en-US-RogerNeural", "name": "Roger (Strong Bass)", "gender": "Male"}
]

def get_best_model():
    """Dynamically finds a working Gemini model."""
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        for preferred in ["models/gemini-1.5-flash", "models/gemini-1.5-flash-001"]:
            if preferred in available_models: return preferred
        return available_models[0] if available_models else "models/gemini-pro"
    except Exception: return "models/gemini-1.5-flash"

WORKING_MODEL_NAME = get_best_model()

@app.on_event("startup")
async def startup_event():
    global sprout_engine, chat_session
    print("🌱 STARTING SPROUT LIFE SERVER...")
    
    # 1. Initialize Engine
    checkpoint = f"{base_dir}/Wav2Lip/checkpoints/wav2lip_gan.pth"
    try:
        sprout_engine = SproutEngine(checkpoint, avatars_dir)
    except Exception as e:
        print(f"❌ Engine Start Failed: {e}")

    # 2. Initialize Memory
    try:
        model = genai.GenerativeModel(
            WORKING_MODEL_NAME,
            system_instruction="You are Sprout, a helpful and fast female AI tutor. Answer in 1 short sentence (max 20 words)."
        )
        chat_session = model.start_chat(history=[])
        print("✅ MEMORY & STATIC FILES READY.")
    except Exception as e:
        print(f"❌ Memory Init Failed: {e}")

@app.get("/config")
def get_config():
    if not sprout_engine: return {"avatars": [], "voices": VOICES}
    avatars = list(sprout_engine.avatar_cache.keys())
    # Sort to ensure womantutor.jpg is first
    avatars.sort(key=lambda x: 0 if x == "womantutor.jpg" else 1)
    return {"avatars": avatars, "voices": VOICES}

@app.get("/reset-memory")
def reset_memory():
    global chat_session
    model = genai.GenerativeModel(WORKING_MODEL_NAME, system_instruction="You are Sprout. Answer short.")
    chat_session = model.start_chat(history=[])
    return {"message": "Memory cleared"}

@app.post("/chat")
async def chat(user_query: str, avatar_id: str = None, voice_id: str = "en-US-JennyNeural"):
    if not sprout_engine: raise HTTPException(500, "Engine not active")
    if not chat_session: raise HTTPException(500, "Memory not initialized")
    if not avatar_id: avatar_id = list(sprout_engine.avatar_cache.keys())[0]

    try:
        prompt = user_query + "\n\nIMPORTANT: You must respond in valid JSON format ONLY. Do not include markdown code blocks. Structure: {\"text\": \"your 1-sentence response\", \"concepts\": [{\"title\": \"Key Term\", \"explanation\": \"1-sentence detail\"}]}"
        
        response = chat_session.send_message(prompt)
        raw_text = response.text
        
        # Clean and parse JSON
        try:
            # Strip markdown if Gemini includes it
            clean_json = re.sub(r'```json\n?|\n?```', '', raw_text).strip()
            data = json.loads(clean_json)
            ai_text = data.get("text", "I'm processing that.")
            concepts = data.get("concepts", [])
        except Exception:
            ai_text = raw_text # Fallback
            concepts = []
        
        # TTS
        audio_path = f"{base_dir}/response.mp3"
        communicate = edge_tts.Communicate(ai_text, voice_id)
        await communicate.save(audio_path)
        
        # Video
        output_path = f"{base_dir}/output_video.mp4"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, sprout_engine.infer, audio_path, output_path, avatar_id)
        
        return {
            "text": ai_text, 
            "concepts": concepts,
            "video_url": "http://127.0.0.1:8000/get-video"
        }
    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise HTTPException(500, str(e))

@app.get("/get-video")
def get_video():
    path = f"{base_dir}/output_video.mp4"
    if os.path.exists(path): return FileResponse(path, media_type="video/mp4")
    raise HTTPException(404, "Video not found")

@app.post("/pdf-to-video")
async def pdf_to_video(file: UploadFile = File(...), avatar_id: str = None, voice_id: str = "en-US-JennyNeural"):
    if not sprout_engine: raise HTTPException(500, "Engine not active")
    if not sprout_engine.avatar_cache: raise HTTPException(500, "No avatars available")
    
    available_avatars = list(sprout_engine.avatar_cache.keys())
    if not avatar_id: avatar_id = available_avatars[0]

    try:
        # Save temp PDF
        temp_pdf = f"{base_dir}/temp_upload.pdf"
        with open(temp_pdf, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract Text
        doc = fitz.open(temp_pdf)
        extracted_text = ""
        for page in doc:
            extracted_text += page.get_text()
        doc.close()

        if not extracted_text.strip():
            raise HTTPException(400, "PDF contains no readable text.")

        # Script Generation via Gemini
        prompt = f"I am uploading a new document. Please read it and respond in valid JSON format ONLY. Structure: {{\"text\": \"2-3 sentence engaging summary script\", \"concepts\": [{{ \"title\": \"Term\", \"explanation\": \"Detail\" }}]}}. Document summary:\n\n{extracted_text[:8000]}"
        
        if not chat_session: raise HTTPException(500, "Memory not initialized")
        response = chat_session.send_message(prompt)
        raw_text = response.text
        
        # Clean and parse JSON
        try:
            clean_json = re.sub(r'```json\n?|\n?```', '', raw_text).strip()
            data = json.loads(clean_json)
            ai_script = data.get("text", "Here is a summary of your document.")
            concepts = data.get("concepts", [])
        except Exception:
            ai_script = raw_text
            concepts = []
        
        # TTS Audio
        audio_path = f"{base_dir}/response.mp3"
        communicate = edge_tts.Communicate(ai_script, voice_id)
        await communicate.save(audio_path)
        
        # Audio-driven Avatar Generation
        output_path = f"{base_dir}/output_video.mp4"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, sprout_engine.infer, audio_path, output_path, avatar_id)
        
        return {
            "text": ai_script, 
            "concepts": concepts,
            "video_url": "http://127.0.0.1:8000/get-video"
        }
    except Exception as e:
        print(f"❌ PDF ERROR: {e}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)