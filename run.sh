#!/bin/bash
echo "Starting Food Suitability Advisor..."

# Start Ollama if not running
systemctl is-active --quiet ollama || ollama serve &
sleep 2

# Start API in background
uvicorn api.main:app --port 9000 &
sleep 5

# Start UI
streamlit run ui/app.py
