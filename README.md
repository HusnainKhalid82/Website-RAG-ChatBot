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

### Run the FastAPI backend

```bash
python3 app.py serve --host 127.0.0.1 --port 8000
```

### Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

### Ask a question (single-shot)

```bash
python3 app.py chat --site-url https://example.com --question "What is this website about?"
```

### Ask questions interactively (keeps chat history for follow-ups)

```bash
python3 app.py chat --site-url https://example.com
# You: What products are offered?
# You: what about pricing?      <- follow-up resolved using chat history
# You: exit
```

## Architecture — the LangGraph RAG flow

The core of the project is a `StateGraph` with conditional routing and a retry
loop (not a linear `retrieve -> generate` chain):

```
              START
                |
                v
           [ retrieve ]  <-------------------+
                |                             |
                v                             |
        [ grade_documents ]                  |
                |                             |
      relevant? |  (conditional edge)        |
       +--------+--------+                    |
       | yes             | no                 |
       v                 v                    |
  [ generate ]     retries left?              |
       |            +---------+               |
       v            | yes     | no            |
      END           v         v               |
             [ rewrite_query ] [ fallback ]   |
                    |               |          |
                    +---- back -----|----------+
                                    v
                                   END
```

- **retrieve** — top-k (k=3) similarity search over the Chroma vector store.
- **grade_documents** — the LLM judges whether the retrieved chunks are actually
  relevant (YES/NO + rationale). This is what prevents answering from noise.
- **rewrite_query** — if not relevant, the LLM rewrites the question to be more
  specific and self-contained (using chat history), then retrieval retries. Max 2 retries.
- **generate** — answers grounded only in the retrieved chunks, citing sources as `[1]`, `[2]`.
- **fallback** — if nothing relevant is found after retries, it says so honestly instead of hallucinating.

## Design notes

- **Loading & cleaning:** each in-domain page is fetched and parsed with BeautifulSoup;
  `nav`, `footer`, `header`, `aside`, `form`, `script`, and `style` tags are stripped so
  boilerplate does not pollute the chunks. `WebBaseLoader` is kept as a per-URL fallback.
- **Chunking:** `RecursiveCharacterTextSplitter` with **chunk_size=1000, overlap=200**.
  1000 characters is roughly a paragraph or two — enough context for a grounded answer
  without diluting the embedding; the 200-char overlap keeps a sentence from being split
  away from its supporting context. Recursive splitting (unlike plain `CharacterTextSplitter`)
  still produces well-sized chunks even when the cleaned text has no blank-line breaks.
- **Retry logic:** the graph grades retrieved docs, rewrites the query up to 2 times, and
  falls back honestly when relevance stays low.
- **Conversation memory:** chat history is threaded through the graph state and stored per
  `conversation_id`, so follow-up questions ("what about pricing?") are resolved in context.
- **Interfaces:** an interactive/one-shot **CLI** (`app.py chat`), a **FastAPI** backend
  (`app.py serve`), and a **Streamlit** UI (`streamlit run streamlit_app.py`).

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
