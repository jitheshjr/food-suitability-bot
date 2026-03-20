import ollama
import os

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")

def generate_response(prompt: str) -> str:
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful diet advisor. Answer in plain English. Be direct and concise.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.2,
                'num_predict': 180,
                'top_p': 0.9,
                'stop': ['\n\n\n', 'PATIENT:', 'FOOD:', 'KEY RISK']
            }
        )
        return response['message']['content'].strip()

    except Exception as e:
        return (f"Unable to generate response. Error: {str(e)}. "
                f"Please ensure Ollama is running.")
