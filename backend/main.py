import os
import uvicorn
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

if not api_key:
    print("CRITICAL ERROR: GEMINI_API_KEY not found.")
else:
    genai.configure(api_key=api_key)
    print("API Key configured.")

# 2. Dynamic Model Selection Logic
# This function finds ANY working model on your account
def get_working_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"DEBUG: Found working model: {m.name}")
                return m.name
    except Exception as e:
        print(f"Error listing models: {e}")
    return "models/gemini-1.5-flash-latest" # Fallback

WORKING_MODEL_NAME = get_working_model()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def generate_audio(text, output_file="response.mp3"):
    print(f"Generating audio...")
    communicate = edge_tts.Communicate(text, "en-US-AnaNeural")
    await communicate.save(output_file)
    return output_file

@app.get("/")
def home():
    return {"message": f"Sprout AI Engine using {WORKING_MODEL_NAME}"}

@app.post("/chat")
async def chat_with_sprout(user_query: str):
    print(f"\n--- Request: {user_query} ---")
    try:
        # Use the model name we found dynamically
        model = genai.GenerativeModel(WORKING_MODEL_NAME)
        
        prompt = f"You are Sprout, a friendly tutor. Answer in 2 short sentences: {user_query}"
        response = model.generate_content(prompt)
        
        ai_text = response.text
        print(f"AI Text: {ai_text}")
        
        await generate_audio(ai_text)
        
        return {
            "text": ai_text,
            "audio_url": "http://127.0.0.1:8000/get-audio" 
        }
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-audio")
def get_audio():
    audio_path = os.path.join(os.getcwd(), "response.mp3")
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="No audio file")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)