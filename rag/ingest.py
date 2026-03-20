import os
import sys

# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "..", "data", "rag_documents")
STORE_DIR = os.path.join(BASE_DIR, "vectorstore")

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_all_documents(docs_dir):
    all_docs = []
    files = os.listdir(docs_dir)
    print(f"Found {len(files)} files in {docs_dir}")

    for filename in files:
        filepath = os.path.join(docs_dir, filename)
        print(f"  Loading: {filename}")

        try:
            if filename.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            else:
                print(f"  Skipping unsupported format: {filename}")
                continue

            docs = loader.load()
            # Tag each doc with its source filename
            for doc in docs:
                doc.metadata["source"] = filename
            all_docs.extend(docs)
            print(f"  Loaded {len(docs)} page(s) from {filename}")

        except Exception as e:
            print(f"  ERROR loading {filename}: {e}")

    return all_docs

def main():
    print("="*50)
    print("RAG INGESTION PIPELINE")
    print("="*50)

    # Step 1 — Load documents
    print("\n[1/4] Loading documents...")
    docs = load_all_documents(DOCS_DIR)
    print(f"Total pages loaded: {len(docs)}")

    if len(docs) == 0:
        print("ERROR: No documents loaded. Check your rag_documents folder.")
        sys.exit(1)

    # Step 2 — Split into chunks
    print("\n[2/4] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    print(f"Sample chunk:\n---\n{chunks[0].page_content[:200]}\n---")

    # Step 3 — Load embedding model
    print("\n[3/4] Loading embedding model (downloads ~80MB on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Embedding model loaded.")

    # Step 4 — Build and persist ChromaDB
    print("\n[4/4] Building vector store...")
    # Remove old store if exists
    if os.path.exists(STORE_DIR):
        import shutil
        shutil.rmtree(STORE_DIR)
        print("Removed old vector store.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=STORE_DIR
    )
    vectorstore.persist()

    count = vectorstore._collection.count()
    print(f"Vector store built with {count} vectors.")
    print(f"Saved to: {STORE_DIR}")

    print("\n" + "="*50)
    print("INGESTION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()
