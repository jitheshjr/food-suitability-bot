## Food Suitability Advisor

An AI-powered conversational chatbot that helps patients with chronic 
health conditions — diabetes, hypertension, and kidney disease — 
determine whether a specific food is safe, moderate, or unsuitable 
for their condition.

The system combines four AI components into a single pipeline:
- A RAG retriever that grounds responses in real medical guidelines
  (WHO, ADA, NHLBI) using ChromaDB and MiniLM embeddings
- An XGBoost classifier trained on 8,000 patient-food records that
  predicts food suitability with 0.985 macro F1
- SHAP explainability that exposes the top contributing nutritional
  risk factors behind every prediction
- Groq LLM (Llama 3.1 8B) that generates a natural language
  explanation — while Python deterministically controls the verdict

A conversational session manager collects missing information
(age, condition, food) across multiple turns before triggering
the pipeline, handling incomplete queries, gibberish, and
off-topic inputs gracefully.

Built entirely on free tools. Runs on 6GB RAM with no GPU.
