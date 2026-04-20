#!/bin/bash
echo "Starting Food Suitability Advisor..."

# Start API in background
uvicorn api.main:app --port 8000 &
API_PID=$!
echo "API started (PID: $API_PID)"
sleep 4

# Start UI (foreground — keeps script alive)
streamlit run ui/app.py

# When streamlit exits, kill the API too
kill $API_PID 2>/dev/null
echo "Stopped."