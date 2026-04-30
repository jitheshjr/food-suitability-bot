# Food Suitability Advisor

Food Suitability Advisor is a prototype chatbot that helps answer questions such as:

> Can I eat ice cream if I have diabetes?

It combines conversational slot filling, local nutrition lookup, retrieval-augmented diet guidance, an XGBoost suitability model, SHAP explanations, and a Groq-powered language model response.

This is an academic/demo project. It is not clinically validated and must not be used as a substitute for advice from a doctor, dietitian, or other qualified healthcare professional.

## Features

- Multi-turn chat flow that collects required details such as food, age, and health condition
- Local food nutrition lookup from processed CSV data
- RAG retrieval from curated diet guideline documents
- ML prediction into `safe`, `moderate`, or `avoid`
- SHAP-based reason summaries for model behavior
- FastAPI backend and Streamlit frontend
- Unit tests for session handling, food lookup, and response formatting

## Project Structure

```text
.
├── api/                  # FastAPI application and chat endpoints
├── chatbot/              # Conversation pipeline, entity extraction, fusion, lookup
├── data/                 # Processed datasets, source documents, preprocessing scripts
├── ml_model/             # Training, prediction, SHAP explanation, saved artifacts
├── notebooks/            # Exploration and model development notebooks
├── rag/                  # Document ingestion, Chroma retrieval, retrieval evaluation
├── tests/                # Unit tests
├── ui/                   # Streamlit chat frontend
├── requirements.txt      # Python dependencies
└── run.sh                # Starts API and UI together
```

## How It Works

1. A user sends a message through Streamlit or the FastAPI `/chat` endpoint.
2. The session manager extracts and stores conversation fields across turns.
3. Once enough information is available, the pipeline:
   - looks up the requested food in the local nutrition table
   - retrieves relevant guideline passages from the vector store
   - predicts suitability with the trained XGBoost model
   - generates SHAP-based model reasons
   - asks the LLM to produce a plain-language explanation
4. The final response includes the verdict, confidence, reasoning, and source filenames used as evidence.

## Requirements

- Python 3.11+
- Groq API key
- Local model artifacts in `ml_model/model/`
- RAG vector store in `rag/vectorstore/`, or source documents available so it can be rebuilt

## Setup

Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

`GROQ_MODEL` is optional. If it is not set, the app uses `llama-3.1-8b-instant`.

## Running The App

Start the API and Streamlit UI together:

```bash
./run.sh
```

Default URLs:

- API: `http://localhost:8000`
- UI: `http://localhost:8501`

You can also run each service separately:

```bash
uvicorn api.main:app --reload --port 8000
```

```bash
streamlit run ui/app.py
```

## API Usage

Health check:

```bash
curl http://localhost:8000/health
```

Send a chat turn:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-session",
    "message": "I am 64 and diabetic. Can I eat ice cream?"
  }'
```

Useful endpoints:

- `GET /` returns API status metadata
- `GET /health` returns a simple health check
- `POST /chat` sends a conversation turn
- `POST /reset/{session_id}` clears one in-memory session
- `GET /session/{session_id}` returns session state for debugging

## Rebuilding Assets

Rebuild food and labeled datasets:

```bash
python data/preprocess_foods.py
python data/merge_gi.py
python data/generate_dataset.py
```

Retrain the ML model:

```bash
cd ml_model
python train.py
cd ..
```

Rebuild SHAP plots:

```bash
cd ml_model
python explain.py
cd ..
```

Rebuild the RAG vector store:

```bash
python rag/ingest.py
```

Evaluate retrieval:

```bash
python rag/evaluate_retrieval.py
```

## Tests

Run the unit test suite:

```bash
python -m unittest discover -s tests -v
```

The current tests cover:

- multi-turn session flow
- food lookup success and fallback behavior
- final response source formatting

## Known Limitations

- Suitability labels are generated from synthetic rules, not clinician-labeled real-world outcomes.
- Session state is stored in memory and resets when the API process restarts.
- Food matching is heuristic and may fail for uncommon foods, brand names, or complex dishes.
- The system currently handles a narrow set of health conditions and nutrition features.
- Deployment requires the local model files, processed datasets, and vector store assets to be present.
- Generated guidance can be incomplete or wrong and should be reviewed by a qualified professional before use.

## Suggested Improvements

- Add portion size handling
- Support multiple simultaneous conditions
- Improve food aliases and dish decomposition
- Store sessions in Redis or a database
- Return structured citations separately from the generated answer
- Add end-to-end tests with mocked LLM calls
- Add deployment configuration for production environments
