# Website RAG Chatbot

A minimal retrieval-augmented generation bot for website content using LangChain and LangGraph.

## What is included

- `ingest.py` — crawls a website, cleans text, chunks pages, and stores embeddings in Chroma.
- `rag_bot.py` — builds a LangGraph with retrieve, grade_documents, rewrite_query, generate, and fallback nodes.
- `app.py` — FastAPI server and CLI wrapper for ingestion and chat.
- `.env.example` — environment variables required to run the bot.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
3. Create a `.env` file from `.env.example` and set provider credentials.

## Usage

### Using Gemini

Set the provider environment variables in `.env`:

```bash
AI_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
```

### Ingest a website

```bash
python3 app.py ingest --site-url https://example.com --max-pages 3
```

### Run the FastAPI server

```bash
python3 app.py serve --host 127.0.0.1 --port 8000
```

### Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

### Ask a question

```bash
python3 app.py chat --site-url https://example.com --question "What is this website about?"
```

## Design notes

- The ingestion pipeline uses `WebBaseLoader` and a simple in-domain URL collector.
- Chunking uses `CharacterTextSplitter` with 1000 characters and 200 characters overlap.
- The LangGraph flow grades retrieved docs, rewrites the query up to 2 retries, and falls back honestly when relevance is low.
- Conversation context is preserved in memory for follow-up questions.

## Assignment deliverables

The repository includes the following submission artifacts:

- `README.md` — setup and implementation overview
- `day1_notes.md` — LangChain and LangGraph learning notes with proof-of-understanding
- `langgraph_toy_graph.py` — a toy StateGraph example with conditional routing
- `submission_writeup.md` — write-up covering chunking strategy, failure cases, and production improvements
- `requirements.txt` — dependency list

## Slack upload guidance

For the Slack submission, upload:

- `README.md`
- `day1_notes.md`
- `langgraph_toy_graph.py`
- `submission_writeup.md`
- `requirements.txt`

Do not upload `.env`, `.env.example`, `data/`, or any local vector store files.

## Notes

- Do not commit `.env` or `data/`.
