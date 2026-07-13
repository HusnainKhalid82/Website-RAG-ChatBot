import logging
import os
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Union

from langchain.schema import AIMessage, Document, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from ingest import get_vector_store

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
TOP_K = 3


def parse_yes_no(answer: str) -> bool:
    normalized = answer.strip().lower()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0]
    if first_line.startswith("yes") and "no" not in first_line:
        return True
    if "yes" in first_line and "no" not in first_line:
        return True
    return False


def messages_from_history(history: List[Dict[str, str]]) -> List[Union[HumanMessage, AIMessage]]:
    messages: List[Union[HumanMessage, AIMessage]] = []
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def build_rag_graph(retriever: Any, llm: ChatOpenAI, max_retries: int = MAX_RETRIES) -> Any:
    graph = StateGraph(dict)

    def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("question", "")
        docs = retriever.get_relevant_documents(query)
        state["retrieved_docs"] = docs
        state["retrieved_count"] = len(docs)
        logger.debug("Retrieved %s docs for query=%s", len(docs), query)
        return state

    def grade_documents(state: Dict[str, Any]) -> Dict[str, Any]:
        docs: List[Document] = state.get("retrieved_docs", []) or []
        if not docs:
            state["relevant"] = False
            state["relevance_reason"] = "No documents were retrieved."
            return state

        prompt = [
            SystemMessage(
                content=(
                    "You are a relevance judge. Read the question and the retrieved "
                    "document excerpts, then decide whether these excerpts are "
                    "relevant enough to answer the question accurately."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{state.get('question')}\n\n"
                    "Document excerpts:\n"
                    + "\n---\n".join(
                        f"Source: {doc.metadata.get('source_url', doc.metadata.get('source', 'unknown'))}\n{doc.page_content[:1000]}"
                        for doc in docs
                    )
                    + "\n\nAnswer only with YES or NO and a short rationale."
                )
            ),
        ]
        result = llm.predict_messages(prompt).content.strip()
        state["relevant"] = parse_yes_no(result)
        state["relevance_reason"] = result
        logger.debug("Grade documents result=%s", result)
        return state

    def rewrite_query(state: Dict[str, Any]) -> Dict[str, Any]:
        previous_query = state.get("question", "")
        history = state.get("history", []) or []
        conversation = "\n".join(
            f"User: {item['content']}" if item["role"] == "user" else f"Assistant: {item['content']}"
            for item in history
        )
        prompt = [
            SystemMessage(
                content=(
                    "You are a question rewriting assistant. Rewrite the user's most recent "
                    "question so it is more specific and focused for document retrieval, "
                    "while preserving the original intent."
                )
            ),
            HumanMessage(
                content=(
                    f"Conversation history:\n{conversation}\n\n"
                    f"Original question:\n{previous_query}\n\n"
                    "If the user asked a follow-up, make the rewritten query self-contained. "
                    "Return only the rewritten query."
                )
            ),
        ]
        rewritten = llm.predict_messages(prompt).content.strip()
        state["attempt"] = state.get("attempt", 0) + 1
        state["question"] = rewritten or previous_query
        state["rewrite_reason"] = rewritten
        logger.debug("Rewrote query=%s to %s", previous_query, rewritten)
        return state

    def generate(state: Dict[str, Any]) -> Dict[str, Any]:
        docs: List[Document] = state.get("retrieved_docs", []) or []
        question = state.get("question", "")
        history = state.get("history", []) or []
        source_blocks = []
        for index, doc in enumerate(docs, start=1):
            source_blocks.append(
                f"[{index}] {doc.metadata.get('source_url', doc.metadata.get('source', 'unknown'))}\n{doc.page_content}"
            )
        prompt = [
            SystemMessage(content=(
                "You are a helpful assistant that answers questions using only the "
                "provided website excerpts. Cite every fact with source numbers like [1]. "
                "If the answer is not found, say you could not find enough information."
            )),
            HumanMessage(
                content=(
                    f"Question:\n{question}\n\n"
                    "Website excerpts:\n"
                    + "\n\n".join(source_blocks)
                    + "\n\n"
                    "Conversation history:\n"
                    + ("\n".join(
                        f"User: {item['content']}" if item['role'] == "user" else f"Assistant: {item['content']}"
                        for item in history
                    ) or "None")
                    + "\n\nProvide a final answer grounded in the excerpts."
                )
            ),
        ]
        answer = llm.predict_messages(prompt).content.strip()
        state["answer"] = answer
        state["source_citations"] = [doc.metadata.get("source_url", doc.metadata.get("source", "unknown")) for doc in docs]
        logger.debug("Generated answer length=%s", len(answer))
        return state

    def fallback(state: Dict[str, Any]) -> Dict[str, Any]:
        state["answer"] = (
            "I could not find a reliable answer in the ingested website content. "
            "Please try a different question or ingest a site with more information."
        )
        state["source_citations"] = []
        return state

    def choose_next_node(state: Dict[str, Any]) -> str:
        if state.get("relevant"):
            return "generate"
        if state.get("attempt", 0) >= max_retries:
            return "fallback"
        return "rewrite_query"

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", choose_next_node)
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    return graph.compile()


class GeminiChat:
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.2):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature

    def predict_messages(self, messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) -> Any:
        payload = []
        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, AIMessage):
                role = "model"
            else:
                role = "user"
            payload.append({"role": role, "parts": [message.content]})

        response = self.model.generate_content(payload)
        text = getattr(response, "text", None)
        if not text and getattr(response, "candidates", None):
            candidate = response.candidates[0]
            content = getattr(candidate, "content", None)
            if content is not None and getattr(content, "parts", None):
                text = "".join(getattr(part, "text", "") for part in content.parts)
        return SimpleNamespace(content=text or "")


def get_llm():
    provider = os.getenv("AI_PROVIDER", "openai").strip().lower()
    if provider == "gemini":
        return GeminiChat(
            model_name=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.2,
        )
    return ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


class SiteRAGBot:
    def __init__(self, persist_directory: str = "data/vectordb"):
        self.persist_directory = persist_directory
        self.llm = get_llm()
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def get_retriever(self, site_url: str) -> Any:
        vector_store = get_vector_store(site_url, persist_directory=self.persist_directory)
        return vector_store.as_retriever(search_kwargs={"k": TOP_K})

    def answer_question(
        self,
        site_url: str,
        question: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        conversation_id = conversation_id or site_url
        history = self.sessions.get(conversation_id, [])
        state = {
            "site_url": site_url,
            "question": question,
            "history": history,
            "attempt": 0,
        }
        graph = build_rag_graph(self.get_retriever(site_url), self.llm)
        result = graph.invoke(state)
        user_message = {"role": "user", "content": question}
        assistant_message = {"role": "assistant", "content": result.get("answer", "")}
        updated_history = history + [user_message, assistant_message]
        self.sessions[conversation_id] = updated_history[-10:]
        return {
            "answer": result.get("answer", ""),
            "source_citations": result.get("source_citations", []),
            "relevance": result.get("relevant", False),
            "attempt": result.get("attempt", 0),
            "history": self.sessions[conversation_id],
            "relevance_reason": result.get("relevance_reason", ""),
        }

    def reset_conversation(self, conversation_id: str) -> None:
        self.sessions.pop(conversation_id, None)
