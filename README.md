# Food Suitability Advisor

Food Suitability Advisor is a health-focused chatbot that answers questions like "Can I eat ice cream if I have diabetes?" by combining:

- conversational slot filling for `food`, `age`, and `condition`
- local nutrition lookup
- retrieval-augmented guideline context from curated diet documents
- an XGBoost classifier with SHAP-based feature explanations
- an LLM-generated plain-language explanation

The current repository is best understood as a prototype or academic/demo project rather than a clinically validated medical tool.

## Project Structure

- `api/`: FastAPI backend
- `ui/`: Streamlit chat frontend
- `chatbot/`: session flow, extraction, response fusion, food lookup
- `rag/`: ingestion, retrieval, and retrieval evaluation
- `ml_model/`: training, prediction, and SHAP explanation utilities
- `data/`: processed nutrition data, RAG source docs, and dataset generation scripts
- `notebooks/`: exploratory notebooks used during development

## How It Works

1. The user sends a question through the Streamlit UI or the API.
2. The chatbot extracts `food`, `age`, and `condition` and stores partial progress across turns.
3. Once all required fields are present, the pipeline:
   - finds nutrition values for the requested food
   - retrieves relevant guideline passages from the vector store
   - predicts a `safe`, `moderate`, or `avoid` label
   - generates top SHAP reasons
   - asks the LLM to explain the decision in plain English
4. The final answer includes the verdict, explanation, key factors, and visible RAG source citations.

## Requirements

- Python 3.11+ recommended
- a Groq API key for entity extraction and final explanation

## Environment Variables

Create a `.env` file in the project root with:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

`GROQ_MODEL` is optional. If omitted, the default model above is used.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Running The App

Start both the API and Streamlit UI:

```bash
./run.sh
```

Or run them separately:

```bash
uvicorn api.main:app --reload --port 8000
```

```bash
streamlit run ui/app.py
```

Default URLs:

- API: `http://localhost:8000`
- UI: `http://localhost:8501`

## Rebuilding Data Assets

Rebuild the food tables or synthetic labeled dataset if needed:

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

Evaluate retrieval quality:

```bash
python rag/evaluate_retrieval.py
```

## Running Tests

This repository includes lightweight unit tests for core behavior:

```bash
python -m unittest discover -s tests -v
```

The tests currently focus on:

- session flow and multi-turn field collection
- citation formatting in the final answer
- food lookup success and fallback behavior

## API Overview

### `GET /`

Returns API status metadata.

### `GET /health`

Returns a simple health check.

### `POST /chat`

Request body:

```json
{
  "session_id": "demo-session",
  "message": "I am 64 and diabetic. Can I eat ice cream?"
}
```

### `POST /reset/{session_id}`

Clears the current conversation state for the provided session.

### `GET /session/{session_id}`

Returns the current in-memory session state.

## Known Limitations

- The ML labels are generated from synthetic rules rather than clinician-labeled real-world data.
- Session state is stored in memory only.
- Food matching is heuristic and may fall back to default nutrition when a match is weak.
- The tool is not a substitute for professional medical advice.

## Suggested Next Improvements

- support portion sizes and multiple conditions
- improve food alias matching and dish decomposition
- persist sessions with Redis or a database
- expose structured citations separately in the API response
- add end-to-end tests around the full pipeline with mocked LLM calls
