import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot.pipeline import run_pipeline

app = FastAPI(
    title="Food Suitability Advisor API",
    description="Domain-specific chatbot for food health guidance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response:      str
    ml_label:      str
    ml_confidence: float
    entities:      dict
    shap_reasons:  list

@app.get("/")
def root():
    return {"status": "running", "model": "tinyllama"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        result = run_pipeline(request.query)
        return QueryResponse(
            response      = result['response'],
            ml_label      = result['ml_label'],
            ml_confidence = result['ml_confidence'],
            entities      = result['entities'],
            shap_reasons  = result['shap_reasons'],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
