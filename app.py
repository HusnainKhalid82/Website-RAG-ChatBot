import argparse
import logging
import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
import uvicorn

from ingest import ingest_site
from rag_bot import SiteRAGBot

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Website RAG Chatbot",
    description="Ingest a website, then ask questions grounded in the ingested site content.",
    version="0.1.0",
)

static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

bot = SiteRAGBot(persist_directory=os.getenv("VECTORDIR", "data/vectordb"))


class IngestRequest(BaseModel):
    site_url: HttpUrl
    max_pages: int = 3


class ChatRequest(BaseModel):
    site_url: HttpUrl
    question: str
    conversation_id: Optional[str] = None


class ResetRequest(BaseModel):
    conversation_id: str


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    try:
        result = ingest_site(str(request.site_url), max_pages=request.max_pages, persist_directory=bot.persist_directory)
        return {"status": "ok", "detail": result}
    except Exception as exc:
        logger.exception("Failed ingesting site %s", request.site_url)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    try:
        result = bot.answer_question(str(request.site_url), request.question, request.conversation_id)
        return {"status": "ok", "result": result}
    except Exception as exc:
        logger.exception("Failed answering question for site %s", request.site_url)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/reset")
def reset(request: ResetRequest) -> dict:
    bot.reset_conversation(request.conversation_id)
    return {"status": "ok", "conversation_id": request.conversation_id}


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "message": "RAG chatbot is ready."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Website RAG Chatbot server and CLI")
    parser.add_argument("command", choices=["serve", "ingest", "chat"], help="Action to run")
    parser.add_argument("--site-url", help="Website URL to ingest or ask about")
    parser.add_argument("--max-pages", type=int, default=3, help="Maximum number of pages to crawl during ingest")
    parser.add_argument("--question", help="Question to ask the ingested website")
    parser.add_argument("--conversation-id", help="Conversation identifier for follow-up questions")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    if args.command == "serve":
        uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
    elif args.command == "ingest":
        if not args.site_url:
            raise SystemExit("--site-url is required for ingest")
        print("Ingesting site:", args.site_url)
        result = ingest_site(args.site_url, max_pages=args.max_pages, persist_directory=bot.persist_directory)
        print(result)
    elif args.command == "chat":
        if not args.site_url or not args.question:
            raise SystemExit("--site-url and --question are required for chat")
        print("Asking:", args.question)
        result = bot.answer_question(args.site_url, args.question, args.conversation_id)
        print(result)


if __name__ == "__main__":
    main()
