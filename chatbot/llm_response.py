import os
from groq import Groq

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_response(prompt: str) -> str:
    try:
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a helpful diet advisor. '
                        'Answer in plain English. '
                        'Be direct, concise, and accurate. '
                        'Never exceed 3 sentences.'
                    )
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return (
            f"I was unable to generate a response at this time. "
            f"Error: {str(e)}"
        )