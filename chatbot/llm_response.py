import ollama
import os

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")

def generate_response(prompt: str) -> str:
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a clinical diet advisor. '
                        'Answer in plain English. '
                        'Be direct, concise, and factual. '
                        'Never exceed 3 sentences.'
                    )
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.1,
                'num_predict': 200,
                'top_p':       0.9,
                'stop':        ['\n\n\n', 'PATIENT:', 'FOOD:', 'INPUT:', 'Output:']
            }
        )
        return response['message']['content'].strip()

    except Exception as e:
        return (
            f"Unable to generate a response at this time. "
            f"Please ensure Ollama is running. (Error: {str(e)})"
        )