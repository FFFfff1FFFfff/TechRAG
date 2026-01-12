# TechRAG - Course Q&A System

A simple RAG system using OpenAI API to answer questions based on course markdown files.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

Or create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

3. Add your course markdown files to the `docs/` directory.

## Usage

```bash
python main.py
```

### Commands

- Type any question to query the course content
- `reload` - Reload documents after adding new files
- `quit` - Exit the application

## Project Structure

```
TechRAG/
├── main.py           # Main application
├── requirements.txt  # Dependencies
├── docs/             # Place markdown files here
└── .env              # Environment variables (optional)
```
