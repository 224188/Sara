import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone

app = FastAPI(title="SARA - Boss ka Personal AI")

# CORS Setup taaki aage chal kar humara React frontend app isse bina error ke connect ho sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MONGO_URI = os.getenv("MONGO_URI", "")

# Connections Initialization
db_client = AsyncIOMotorClient(MONGO_URI) if MONGO_URI else None
db = db_client["sara_database"] if db_client else None
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.get("/")
async def health_check():
    db_status = "Connected" if db_client else "Missing MONGO_URI"
    ai_status = "Ready" if ai_client else "Missing GEMINI_API_KEY"
    return {
        "status": "healthy",
        "database": db_status,
        "ai_brain": ai_status
    }

@app.post("/chat")
async def chat_with_sara(req: ChatRequest):
    if not ai_client or not db:
        raise HTTPException(status_code=500, detail="SARA ka system poori tarah configured nahi hai, Boss!")
    
    # 1. MongoDB se purani chat history nikaalna (Memory)
    history_cursor = db["chat_history"].find({"session_id": req.session_id}).sort("timestamp", 1)
    history = await history_cursor.to_list(length=20)
    
    # 2. Gemini ke samajhne ke liye format taiyar karna
    formatted_contents = []
    
    # History formats ko correct types structures me convert karna
    for h in history:
        if h.get("user_msg") and h.get("sara_res"):
            formatted_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=h["user_msg"])]))
            formatted_contents.append(types.Content(role="model", parts=[types.Part.from_text(text=h["sara_res"])]))
    
    # Naya message history mein add karna
    formatted_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=req.message)]))
    
    # System Instructions taaki SARA hamesha loyal assistant rahe
    system_instruction = "Aap SARA hain, Boss ke personal AI assistant aur executive trading desk controller. Hamesha loyal, smart aur response Hinglish/Hindi mein short aur punchy rakhein. Har response mein Boss ko zaroor address karein."
    
    try:
        # 3. Gemini Flash AI se jawab mangna (Using Correct Types Configuration)
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=formatted_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        sara_response = response.text
        
        # 4. Nayi baat ko MongoDB memory mein safe UTC timezone ke saath save karna
        await db["chat_history"].insert_one({
            "session_id": req.session_id,
            "user_msg": req.message,
            "sara_res": sara_response,
            "timestamp": datetime.now(timezone.utc)
        })
        
        return {"response": sara_response}
        
    except Exception as e:
        # Render logs mein exact error check karne ke liye print statement
        print(f"Asli AI Error Check Karo, Boss: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")
