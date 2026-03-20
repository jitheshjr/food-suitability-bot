import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# Load once at import time
print("Loading RAG retriever...")
_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
_vectorstore = Chroma(
    persist_directory=STORE_DIR,
    embedding_function=_embeddings
)
print(f"Vector store loaded — {_vectorstore._collection.count()} chunks available.")


def retrieve_context(query: str, k: int = 3) -> list[dict]:
    """
    Returns top-k relevant chunks for the query.
    Each result is a dict: {text, source, score}
    """
    results = _vectorstore.similarity_search_with_relevance_scores(query, k=k)

    retrieved = []
    for doc, score in results:
        retrieved.append({
            "text":   doc.page_content.strip(),
            "source": doc.metadata.get("source", "unknown"),
            "score":  round(score, 3)
        })

    return retrieved


def format_context_for_prompt(retrieved: list[dict]) -> str:
    """
    Formats retrieved chunks into a clean string for the LLM prompt.
    """
    lines = []
    for i, item in enumerate(retrieved, 1):
        lines.append(f"[{i}] (source: {item['source']}, relevance: {item['score']})")
        lines.append(item['text'])
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    # Test retrieval
    test_queries = [
        "can a diabetic patient eat ice cream",
        "sodium intake for hypertension",
        "protein restriction kidney disease",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        results = retrieve_context(query, k=3)
        for r in results:
            print(f"Score: {r['score']} | Source: {r['source']}")
            print(f"Text: {r['text'][:150]}...")
            print()
