"""
Simple RAG System for Course Q&A

Uses OpenAI API directly to read MD files and answer questions.
Supports image uploads for visual questions.
"""

import base64
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path("docs")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def load_documents():
    """Load all markdown files from docs directory."""
    if not DOCS_DIR.exists():
        return []

    documents = []
    for md_file in DOCS_DIR.glob("**/*.md"):
        content = md_file.read_text(encoding="utf-8")
        documents.append(f"# File: {md_file.name}\n\n{content}")

    return documents


def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(path):
    """Get media type for image."""
    suffix = Path(path).suffix.lower()
    types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return types.get(suffix, "image/png")


def parse_input(user_input):
    """Parse input to extract image paths and question.

    Images are specified with @ prefix: @/path/to/image.png
    Example: @screenshot.png @diagram.jpg What does this show?
    """
    pattern = r"@(\S+\.(?:png|jpg|jpeg|gif|webp))"
    image_paths = re.findall(pattern, user_input, re.IGNORECASE)
    question = re.sub(pattern, "", user_input, flags=re.IGNORECASE).strip()
    return image_paths, question


def build_user_content(context, question, image_paths):
    """Build user message content with optional images."""
    content = []

    # Add text content first
    if image_paths:
        text = f"Course Materials:\n\n{context}\n\n---\n\nI have attached {len(image_paths)} image(s). Please analyze the image(s) carefully and answer: {question}"
    else:
        text = f"Course Materials:\n\n{context}\n\n---\n\nQuestion: {question}"
    content.append({"type": "text", "text": text})

    # Add images after text
    for img_path in image_paths:
        path = Path(img_path)
        if not path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{get_image_media_type(img_path)};base64,{encode_image(path)}"
            }
        })

    return content


def query(client, documents, question, image_paths=None):
    """Send question with document context and optional images to OpenAI."""
    if not documents:
        print("Error: No markdown files found in 'docs/' directory.")
        return

    image_paths = image_paths or []
    context = "\n\n---\n\n".join(documents)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful teaching assistant. Answer questions based on the provided course materials. When images are provided, you MUST analyze them in detail and describe what you see. Be concise and accurate.",
            },
            {
                "role": "user",
                "content": build_user_content(context, question, image_paths),
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
    print("To attach images: @path/to/image.png your question")
    print("Commands: 'reload', 'quit'")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "reload":
            documents = load_documents()
            print(f"Reloaded {len(documents)} document(s).\n")
            continue

        image_paths, question = parse_input(user_input)

        if not question:
            print("Please enter a question.\n")
            continue

        if image_paths:
            print(f"Attached {len(image_paths)} image(s)")

        query(client, documents, question, image_paths)


if __name__ == "__main__":
    main()
