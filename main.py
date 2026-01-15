"""
Simple RAG System for Course Q&A

Uses OpenAI API directly to read MD files and answer questions.
Supports image uploads for visual questions.
Uses Socratic method to guide students through problems.
"""

import base64
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path("docs")

SYSTEM_PROMPT = """You are a Socratic teaching assistant. Your goal is to help students learn by guiding them to discover answers themselves.

When a student asks a question or shows their work:

1. FIRST, analyze what they've done (especially any images they share)
2. If there's an error or misunderstanding:
   - Do NOT immediately give the correct answer
   - Ask guiding questions to help them identify the issue
   - Give hints that lead them toward the solution
   - Only after 2-3 exchanges of guidance, provide the full explanation
3. If they're correct, confirm and explain why

Use the course materials provided as reference. When analyzing images, describe what you observe in detail.

Keep responses concise but educational. Remember: guide first, answer later."""


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
    """Parse input to extract image paths and question."""
    pattern = r"@(\S+\.(?:png|jpg|jpeg|gif|webp))"
    image_paths = re.findall(pattern, user_input, re.IGNORECASE)
    question = re.sub(pattern, "", user_input, flags=re.IGNORECASE).strip()
    return image_paths, question


def build_user_content(question, image_paths):
    """Build user message content with optional images."""
    content = [{"type": "text", "text": question}]

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


def query(client, messages):
    """Send messages to OpenAI and return response."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
    )

    print("\nA: ", end="", flush=True)
    full_response = []
    for chunk in response:
        if chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            print(text, end="", flush=True)
            full_response.append(text)
    print("\n")

    return "".join(full_response)


def main():
    """Main entry point."""
    client = OpenAI()
    documents = load_documents()
    context = "\n\n---\n\n".join(documents)

    # Conversation history
    history = []

    print(f"Loaded {len(documents)} document(s) from 'docs/'")
    print("\n" + "=" * 50)
    print("Course Q&A System (Socratic Mode)")
    print("Commands:")
    print("  new    - Start new conversation")
    print("  answer <question> - Get direct answer (skip guidance)")
    print("  reload - Reload documents")
    print("  quit   - Exit")
    print("Attach images: @path/to/image.png your question")
    print("=" * 50 + "\n")

    while True:
        try:
            print("Q: ", end="", flush=True)
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        # Debug: show what was received
        if len(user_input) < 20:
            print(f"[DEBUG] Received: '{user_input}' (len={len(user_input)})")

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "reload":
            documents = load_documents()
            context = "\n\n---\n\n".join(documents)
            print(f"Reloaded {len(documents)} document(s).\n")
            continue

        if user_input.lower() == "new":
            history = []
            print("Started new conversation.\n")
            continue

        # Direct answer mode - skip Socratic guidance
        direct_mode = False
        if user_input.lower().startswith("answer "):
            user_input = user_input[7:]  # Remove "answer " prefix
            direct_mode = True

        image_paths, question = parse_input(user_input)

        # Skip if no actual question text (must have at least 2 chars)
        if not question or len(question.strip()) < 2:
            if image_paths:
                print("Please enter a question along with the image.\n")
            continue

        if image_paths:
            print(f"Attached {len(image_paths)} image(s)")

        # Build messages with history
        if direct_mode:
            system_content = f"You are a helpful teaching assistant. Answer the question directly and concisely based on the course materials. Analyze any images in detail.\n\nCourse Materials:\n{context}"
        else:
            system_content = f"{SYSTEM_PROMPT}\n\nCourse Materials:\n{context}"

        messages = [
            {"role": "system", "content": system_content},
        ]
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": build_user_content(question, image_paths),
        })

        # Get response and update history
        response = query(client, messages)

        # Save to history (text only for user, to avoid re-sending images)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})

        # Keep history reasonable (last 10 exchanges)
        if len(history) > 20:
            history = history[-20:]


if __name__ == "__main__":
    main()
