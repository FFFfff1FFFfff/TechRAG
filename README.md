# TechRAG - Course Q&A System

A simple RAG (Retrieval-Augmented Generation) system built with LlamaIndex for indexing course slides and answering questions about homework and course content.

## Features

- Index PDF course slides
- Interactive Q&A interface
- Persistent index storage
- Free local embeddings (HuggingFace)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

3. Add your course PDF files to the `data/` directory.

## Usage

Run the application:

```bash
python main.py
```

### Commands

- Type any question to query the course content
- `rebuild` - Rebuild the index after adding new documents
- `quit` or `exit` - Exit the application

## Project Structure

```
TechRAG/
├── main.py           # Main application
├── requirements.txt  # Dependencies
├── data/             # Place PDF files here
├── storage/          # Index storage (auto-generated)
└── .env              # Environment variables
```

## Notes

- The index is automatically saved to `storage/` and reused on subsequent runs
- To update the index after adding new PDFs, use the `rebuild` command
