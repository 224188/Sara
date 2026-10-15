import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google import genai
from motor.motor_asyncio import AsyncIOMotorClient

app = FastAPI(title="SARA - Boss ka Personal AI")

# Server settings jo bina code badle dashboard se control hongi
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MONGO_URI = os.getenv("MONGO_URI", "")

# MongoDB aur Gemini setup (agar keys dashboard mein hain toh active ho jayenge)
db_client = AsyncIOMotorClient(MONGO_URI) if MONGO_URI else None
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

@app.get("/")
async def health_check():
    # Database aur AI ka status check karne ke liye
    db_status = "Connected" if db_client else "Missing MONGO_URI in Dashboard"
    ai_status = "Ready" if ai_client else "Missing GEMINI_API_KEY in Dashboard"
    
    return {
        "status": "healthy",
        "message": "SARA ka dimaag chalu hai, Boss!",
        "database": db_status,
        "ai_brain": ai_status
    }
