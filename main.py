"""
Simple RAG System for Course Q&A

A minimal RAG implementation using LlamaIndex to index course slides (PDF)
and answer questions about homework and course content.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

# Load environment variables
load_dotenv()

# Paths
DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")


def setup_settings():
    """Configure LlamaIndex settings."""
    # Use HuggingFace embedding model (free, no API key needed)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Use OpenAI for LLM (requires OPENAI_API_KEY in .env)
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)


def build_index():
    """Build a new index from documents in the data directory."""
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        print(f"Error: Please add PDF files to the '{DATA_DIR}' directory.")
        return None

    print("Loading documents...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    print(f"Loaded {len(documents)} document(s).")

    print("Building index...")
    index = VectorStoreIndex.from_documents(documents)

    # Persist index to disk
    STORAGE_DIR.mkdir(exist_ok=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    print(f"Index saved to '{STORAGE_DIR}'.")

    return index


def load_index():
    """Load existing index from storage."""
    if not STORAGE_DIR.exists():
        return None

    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    index = load_index_from_storage(storage_context)
    return index


def get_or_create_index():
    """Load existing index or create a new one."""
    index = load_index()
    if index is None:
        index = build_index()
    return index


def interactive_query(index):
    """Run interactive Q&A session."""
    if index is None:
        return

    query_engine = index.as_query_engine()

    print("\n" + "=" * 50)
    print("Course Q&A System Ready!")
    print("Type your questions about the course or homework.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'rebuild' to rebuild the index.")
    print("=" * 50 + "\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if question.lower() == "rebuild":
            index = build_index()
            if index:
                query_engine = index.as_query_engine()
                print("Index rebuilt successfully.\n")
            continue

        response = query_engine.query(question)
        print(f"\nA: {response}\n")


def main():
    """Main entry point."""
    setup_settings()
    index = get_or_create_index()
    interactive_query(index)


if __name__ == "__main__":
    main()
