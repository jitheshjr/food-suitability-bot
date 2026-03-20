import ollama

response = ollama.chat(
    model='tinyllama',
    messages=[
        {
            'role': 'user',
            'content': 'In one sentence, what should a diabetic patient avoid eating?'
        }
    ]
)

print(response['message']['content'])
