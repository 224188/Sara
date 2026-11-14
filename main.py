import asyncio
import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorClient

from plugin_hub import router as plugin_hub_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sara")

app = FastAPI(title="SARA - Boss ka Personal AI")
app.include_router(plugin_hub_router)

# Public frontend uses JSON requests, not cookie-based authentication.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = "gemini-3.8-flash"
MONGO_URI = os.getenv("MONGO_URI", "").strip()
# Keep MongoDB disabled until its TLS/connection issue is resolved.
# Re-enable later by setting ENABLE_MONGODB=true on the backend.
MONGO_ENABLED = os.getenv("ENABLE_MONGODB", "false").strip().lower() == "true"

db_client = (
    AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    if MONGO_ENABLED and MONGO_URI
    else None
)
db = db_client["sara_database"] if db_client else None
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=10000)


@app.get("/")
async def health_check():
    database_status = "not_configured"
    if db is not None:
        try:
            await db.command("ping")
            database_status = "connected"
        except Exception:
            logger.exception("MongoDB health check failed")
            database_status = "unavailable"

    return {
        "status": "healthy",
        "database": database_status,
        "ai_brain": "ready" if ai_client else "missing_GEMINI_API_KEY",
        "model": GEMINI_MODEL,
    }


@app.post("/chat")
async def chat_with_sara(req: ChatRequest):
    if ai_client is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini is not configured. Check the GEMINI_API_KEY setting on the backend.",
        )

    formatted_contents = []

    # Memory is optional: a MongoDB outage must not prevent SARA from replying.
    if db is not None:
        try:
            cursor = (
                db["chat_history"]
                .find({"session_id": req.session_id})
                .sort("timestamp", -1)
                .limit(20)
            )
            history = await cursor.to_list(length=20)
            for item in reversed(history):
                if item.get("user_msg") and item.get("sara_res"):
                    formatted_contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=item["user_msg"])],
                        )
                    )
                    formatted_contents.append(
                        types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=item["sara_res"])],
                        )
                    )
        except Exception:
            logger.exception("MongoDB history read failed; continuing without memory")

    formatted_contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=req.message)])
    )

    system_instruction = (
        "You are SARA, the personal AI assistant for Boss. "
        "Your personality is loyal, sharp, efficient, and slightly witty. "
        "Address the user as Boss naturally.\n\n"
        "Speak in everyday conversational Hinglish (Hindi + English mix). "
        "Use clear professional English for technical code, core numbers, and trading data.\n\n"
        "DASHBOARD COMMANDS: If Boss explicitly asks to open/show the dashboard, "
        "include exactly: 'Dashboard khol rahi hu, Boss!'. "
        "If Boss explicitly asks to close/hide it, include exactly: "
        "'Dashboard band kar rahi hu, Boss!'. Never say these phrases otherwise.\n\n"
        "SAFETY: Never claim an external action was completed unless it actually was. "
        "Ask Boss for approval before sending, publishing, trading, purchasing, or making "
        "other consequential external changes."
    )

    try:
        # Gemini can temporarily return 503 (high demand) or 429 (rate limit).
        # Retry only transient HTTP statuses; never switch to a paid provider.
        response = None
        max_attempts = 3
        transient_statuses = {429, 500, 502, 503, 504}

        for attempt in range(max_attempts):
            try:
                # The SDK call is synchronous; keep it off FastAPI's event loop.
                response = await asyncio.to_thread(
                    ai_client.models.generate_content,
                    model=GEMINI_MODEL,
                    contents=formatted_contents,
                    config=types.GenerateContentConfig(system_instruction=system_instruction),
                )
                break
            except Exception as exc:
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                try:
                    status = int(status)
                except (TypeError, ValueError):
                    status = None

                if status not in transient_statuses or attempt == max_attempts - 1:
                    logger.exception("Gemini response generation failed")
                    if status == 503:
                        raise HTTPException(
                            status_code=503,
                            detail="Gemini is temporarily busy. Please try again in a little while.",
                        ) from exc
                    if status == 429:
                        raise HTTPException(
                            status_code=429,
                            detail="Gemini free-tier rate limit reached. Please try again later.",
                        ) from exc
                    raise HTTPException(
                        status_code=502,
                        detail="Gemini reply failed. Check backend logs, API key, model access, and free-tier quota.",
                    ) from exc

                wait_seconds = 2 ** attempt
                logger.warning(
                    "Gemini returned transient status %s; retrying in %s seconds (attempt %s/%s)",
                    status,
                    wait_seconds,
                    attempt + 1,
                    max_attempts,
                )
                await asyncio.sleep(wait_seconds)

        sara_response = (response.text or "").strip()
        if not sara_response:
            raise RuntimeError("Gemini returned an empty response")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Gemini response generation failed")
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an invalid or empty reply. Please try again.",
        )

    memory_saved = False
    if db is not None:
        try:
            await db["chat_history"].insert_one(
                {
                    "session_id": req.session_id,
                    "user_msg": req.message,
                    "sara_res": sara_response,
                    "timestamp": datetime.now(timezone.utc),
                }
            )
            memory_saved = True
        except Exception:
            logger.exception("MongoDB memory save failed; returning Gemini reply anyway")

    return {"response": sara_response, "memory_saved": memory_saved}
