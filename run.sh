#!/bin/bash
echo "Starting Food Suitability Advisor..."

# Start Ollama if not running
systemctl is-active --quiet ollama || ollama serve &
sleep 2

# Start API in background
cd ~/Desktop/food_suitability_bot
source env/bin/activate
uvicorn api.main:app --port 8000 &
sleep 5

# Start UI
streamlit run ui/app.py
