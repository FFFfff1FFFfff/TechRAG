"""
Simple RAG System for Course Q&A

Uses OpenAI API directly to read MD files and answer questions.
"""

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path("docs")


def load_documents():
    """Load all markdown files from docs directory."""
    if not DOCS_DIR.exists():
        return []

    documents = []
    for md_file in DOCS_DIR.glob("**/*.md"):
        content = md_file.read_text(encoding="utf-8")
        documents.append(f"# File: {md_file.name}\n\n{content}")

    return documents


def query(client, documents, question):
    """Send question with document context to OpenAI."""
    if not documents:
        print("Error: No markdown files found in 'docs/' directory.")
        return

    context = "\n\n---\n\n".join(documents)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful teaching assistant. Answer questions based on the provided course materials. Be concise and accurate.",
            },
            {
                "role": "user",
                "content": f"Course Materials:\n\n{context}\n\n---\n\nQuestion: {question}",
            },
        ],
        stream=True,
    )

    print("\nA: ", end="", flush=True)
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def main():
    """Main entry point."""
    client = OpenAI()
    documents = load_documents()

    print(f"Loaded {len(documents)} document(s) from 'docs/'")
    print("\n" + "=" * 50)
    print("Course Q&A System Ready!")
    print("Type 'quit' to exit, 'reload' to reload documents.")
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

        if question.lower() == "reload":
            documents = load_documents()
            print(f"Reloaded {len(documents)} document(s).\n")
            continue

        query(client, documents, question)


if __name__ == "__main__":
    main()
