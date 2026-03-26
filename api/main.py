import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from chatbot.pipeline import run_conversation_turn
from chatbot.session_manager import clear_session

app = FastAPI(
    title="Food Suitability Advisor API",
    description="Domain-specific chatbot for food health guidance",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message:    str

class ChatResponse(BaseModel):
    type:           str            # 'message' | 'response' | 'error'
    text:           str
    ml_label:       Optional[str]   = None
    ml_confidence:  Optional[float] = None
    entities:       Optional[dict]  = None
    shap_reasons:   Optional[list]  = None
    session:        dict
    action:         str

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "model": "groq/llama-3.1-8b-instant", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main conversational endpoint.
    Each message needs a session_id to maintain state across turns.
    The same session_id must be sent for all turns in one conversation.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    try:
        result = run_conversation_turn(
            session_id=request.session_id,
            user_text=request.message
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset/{session_id}")
def reset(session_id: str):
    """Clears session state for a given session_id."""
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}

@app.get("/session/{session_id}")
def get_session_info(session_id: str):
    """Returns current session state — useful for debugging."""
    from chatbot.session_manager import get_session
    session = get_session(session_id)
    return session.to_dict()