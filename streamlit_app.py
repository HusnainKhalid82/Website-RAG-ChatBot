import os
from dotenv import load_dotenv
import streamlit as st

from ingest import ingest_site
from rag_bot import SiteRAGBot

load_dotenv()

st.set_page_config(page_title="Website RAG Chatbot", layout="wide")

st.title("Website RAG Chatbot")
st.write("A minimal Streamlit UI for ingesting a website and asking questions.")

site_url = st.text_input("Website URL", value="https://example.com")
max_pages = st.number_input("Max pages to ingest", min_value=1, max_value=10, value=3)

try:
    st.session_state.setdefault("conversation_id", "")
    st.session_state.setdefault("answer", "")
    st.session_state.setdefault("source_citations", [])
    st.session_state.setdefault("status", "Ready")
    st.session_state.setdefault("bot", None)
    st.session_state.setdefault("bot_error", None)
except Exception:
    # session_state is only available when Streamlit runs the script
    pass

provider = os.getenv("AI_PROVIDER", "openai").strip().lower()
required_vars = ["OPENAI_API_KEY"] if provider != "gemini" else ["GEMINI_API_KEY", "GEMINI_MODEL"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    st.warning(f"Missing required environment variables for provider '{provider}': {', '.join(missing_vars)}")
    st.info("Create a .env file from .env.example or export the vars before running Streamlit.")


def get_bot():
    if st.session_state.bot is not None or st.session_state.bot_error is not None:
        return st.session_state.bot
    if missing_vars:
        st.session_state.bot_error = (
            f"Missing required environment variables: {', '.join(missing_vars)}. "
            "Set them in .env or export them before running Streamlit."
        )
        return None
    try:
        st.session_state.bot = SiteRAGBot(persist_directory=os.getenv("VECTORDIR", "data/vectordb"))
    except Exception as exc:
        st.session_state.bot_error = str(exc)
    return st.session_state.bot

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Ingest site")
    if st.button("Ingest website"):
        if not site_url:
            st.session_state.status = "Please enter a valid website URL."
        else:
            bot = get_bot()
            if st.session_state.bot_error:
                st.session_state.status = f"Bot initialization failed: {st.session_state.bot_error}"
            elif bot is None:
                st.session_state.status = "Bot not available."
            else:
                try:
                    st.session_state.status = "Ingesting..."
                    result = ingest_site(site_url, max_pages=max_pages, persist_directory=bot.persist_directory)
                    st.session_state.status = f"Ingest complete: {result['chunks_created']} chunks created from {result['pages_loaded']} pages."
                except Exception as exc:
                    st.session_state.status = f"Ingest error: {exc}"

    st.write(st.session_state.status)

with col2:
    st.subheader("Ask a question")
    question = st.text_area("Question", value="What is this website about?")
    st.session_state.conversation_id = st.text_input("Conversation ID", value=st.session_state.conversation_id)
    if st.button("Ask question"):
        if not site_url or not question:
            st.session_state.status = "Please provide both a website URL and a question."
        else:
            bot = get_bot()
            if st.session_state.bot_error:
                st.session_state.status = f"Bot initialization failed: {st.session_state.bot_error}"
            elif bot is None:
                st.session_state.status = "Bot not available."
            else:
                st.session_state.status = "Querying..."
                try:
                    result = bot.answer_question(site_url, question, st.session_state.conversation_id or site_url)
                    st.session_state.answer = result.get("answer", "")
                    st.session_state.source_citations = result.get("source_citations", [])
                    st.session_state.status = "Answer received."
                except Exception as exc:
                    st.session_state.status = f"Chat error: {exc}"

st.markdown("---")
st.subheader("Answer")
st.write(st.session_state.answer)

st.subheader("Source citations")
if st.session_state.source_citations:
    for source in st.session_state.source_citations:
        st.write(f"- {source}")
else:
    st.write("No sources yet.")
