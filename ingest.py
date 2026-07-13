import os
import re
from pathlib import Path
from typing import List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def sanitize_collection_name(site_url: str) -> str:
    parsed = urlparse(site_url)
    name = parsed.netloc or site_url
    name = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return name.strip("_") or "site"


def same_domain(url: str, base_url: str) -> bool:
    return urlparse(url).netloc == urlparse(base_url).netloc


def collect_site_urls(site_url: str, max_pages: int = 3) -> List[str]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"})
    response = session.get(site_url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    urls = [site_url]
    seen = {site_url}

    for anchor in soup.find_all("a", href=True):
        if len(urls) >= max_pages:
            break
        href = anchor["href"].strip()
        candidate = urljoin(site_url, href)
        if candidate in seen:
            continue
        if candidate.startswith("mailto:") or candidate.startswith("tel:"):
            continue
        if not candidate.startswith(("http://", "https://")):
            continue
        if not same_domain(candidate, site_url):
            continue
        seen.add(candidate)
        urls.append(candidate)

    return urls


# Structural tags that almost never contain answer-bearing content. We strip
# them before extracting text so nav bars, footers, cookie banners, and scripts
# do not pollute the chunks (the "clean the text" step from the brief).
BOILERPLATE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "aside", "form", "noscript", "iframe", "svg",
]


def html_to_clean_text(html: str) -> str:
    """Extract readable text from raw HTML, dropping boilerplate structure."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(BOILERPLATE_TAGS):
        tag.decompose()
    # Prefer the main content region when the page marks one up.
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return main.get_text(separator="\n")


def clean_chunk_text(text: str) -> str:
    """Collapse runs of spaces, drop blank lines, but keep line breaks so the
    recursive splitter still has natural boundaries to split on."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def load_site_documents(site_url: str, max_pages: int = 3) -> List[Document]:
    urls = collect_site_urls(site_url, max_pages=max_pages)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"})

    cleaned_docs: List[Document] = []
    for url in urls:
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            content = clean_chunk_text(html_to_clean_text(response.text))
        except Exception:
            # Fall back to LangChain's WebBaseLoader if the direct fetch fails.
            try:
                loaded = WebBaseLoader(web_paths=[url], show_progress=False).load()
                content = clean_chunk_text(loaded[0].page_content) if loaded else ""
            except Exception:
                content = ""
        if not content:
            continue
        cleaned_docs.append(
            Document(
                page_content=content,
                metadata={"source": url, "source_url": url, "site_url": site_url},
            )
        )
    return cleaned_docs


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    # RecursiveCharacterTextSplitter tries separators in order and falls back to
    # finer ones, so it produces well-sized chunks even when the cleaned text has
    # no blank-line paragraph breaks (CharacterTextSplitter could not).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: List[Document] = []
    for doc in documents:
        text = doc.page_content
        parts = splitter.split_text(text)
        for idx, part in enumerate(parts, start=1):
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = idx
            metadata["source_url"] = metadata.get("source_url", metadata.get("source", ""))
            chunks.append(Document(page_content=part, metadata=metadata))
    return chunks


class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-001"):
        from google.generativeai import configure
        self.api_key = api_key
        self.model_name = model_name
        configure(api_key=self.api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from google.generativeai import embed_content

        # Batch the calls so ingesting a site is a handful of requests instead
        # of one request per chunk (faster and less likely to hit rate limits).
        embeddings: List[List[float]] = []
        batch_size = 100
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            result = embed_content(model=self.model_name, content=batch)
            embeddings.extend(result["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        from google.generativeai import embed_content

        result = embed_content(model=self.model_name, content=text)
        return result["embedding"]


def get_embedding_client():
    provider = os.getenv("AI_PROVIDER", "openai").strip().lower()
    if provider == "gemini":
        return GeminiEmbeddings(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        )
    return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


def get_vector_store(site_url: str, persist_directory: str = "data/vectordb") -> Chroma:
    collection_name = sanitize_collection_name(site_url)
    base_dir = Path(persist_directory)
    base_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_client(),
        persist_directory=str(base_dir),
    )


def create_or_update_vector_store(site_url: str, documents: List[Document], persist_directory: str = "data/vectordb") -> Chroma:
    if not documents:
        raise ValueError("No documents available to ingest.")
    collection_name = sanitize_collection_name(site_url)
    base_dir = Path(persist_directory)
    base_dir.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=get_embedding_client(),
        ids=None,
        collection_name=collection_name,
        persist_directory=str(base_dir),
    )
    # Newer Chroma auto-persists when a persist_directory is set; older versions
    # need an explicit persist(). Support both without crashing.
    if hasattr(vector_store, "persist"):
        try:
            vector_store.persist()
        except Exception:
            pass
    return vector_store


def ingest_site(site_url: str, max_pages: int = 3, persist_directory: str = "data/vectordb") -> dict:
    docs = load_site_documents(site_url, max_pages=max_pages)
    chunks = chunk_documents(docs)
    vector_store = create_or_update_vector_store(site_url, chunks, persist_directory=persist_directory)
    return {
        "site_url": site_url,
        "pages_loaded": len(docs),
        "chunks_created": len(chunks),
        "collection_name": sanitize_collection_name(site_url),
        "persist_directory": str(Path(persist_directory).resolve()),
    }
