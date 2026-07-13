# Day 1 Notes — LangChain + LangGraph Fundamentals

## LangChain fundamentals

### Documents
A `Document` is the basic unit of text that LangChain works with. It stores text and optional metadata like source URLs, page numbers, or chunk identifiers.

### Loaders
Loaders convert raw data sources into `Document` objects. Examples:
- `WebBaseLoader` for crawling web pages
- `TextLoader` for local text files
- `SitemapLoader` for XML sitemaps

### Text Splitters
Text splitters break large documents into smaller chunks for embedding and retrieval. Common splitters are:
- `CharacterTextSplitter`
- `RecursiveCharacterTextSplitter`

Good chunking balances context vs vector size. Too small loses context; too large makes retrieval noisy.

### Embeddings
Embeddings turn text chunks into numeric vectors. Similar text yields similar vectors, which is the foundation of semantic search.

### Vector Stores
Vector stores keep embeddings and metadata. They support similarity search. In this project we use `Chroma`, which stores vectors locally.

### Retrievers
Retrievers query the vector store and return relevant documents. They usually support top-k similarity search and optional metadata filters.

## Tiny LangChain demo script

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from ingest import get_embedding_client  # provider-aware (Gemini or OpenAI)

# 1) Load a single web page
loader = WebBaseLoader("https://example.com")
docs = loader.load()

# 2) Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 3) Embed the chunks and store them in Chroma
embeddings = get_embedding_client()
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./demo_chroma")

# 4) Run a similarity search
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
results = retriever.get_relevant_documents("What is this page about?")
for i, doc in enumerate(results, start=1):
    print(f"Result {i}: {doc.metadata.get('source', 'unknown')}")
    print(doc.page_content[:200])
    print("---")
```

This script proves the basic pipeline: load, split, embed, store, retrieve.

## LangGraph fundamentals

### What is a StateGraph?
A `StateGraph` is a directed graph of processing steps. It keeps a mutable `state` dictionary and runs nodes in order.

### Nodes
Nodes are functions that take the current `state`, perform work, and return an updated `state`.

### Edges
Edges connect nodes. They define the execution flow from one node to the next.

### Conditional edges
Conditional edges let the graph choose the next node based on state values. This is essential for non-linear workflows like retry logic.

## Toy LangGraph example

```python
from langgraph.graph import END, START, StateGraph

# Create a simple graph
graph = StateGraph(dict)

# 1. classify the input
def classify(state):
    text = state.get("text", "").strip().lower()
    state["is_greeting"] = any(greet in text for greet in ["hi", "hello", "hey"])
    return state

# 2. respond to greeting
def respond_greeting(state):
    state["response"] = "Hello! How can I help you today?"
    return state

# 3. respond to question
def respond_question(state):
    state["response"] = "That sounds like a question. I would answer it here."
    return state

# Add nodes
graph.add_node("classify", classify)
graph.add_node("greeting", respond_greeting)
graph.add_node("question", respond_question)

# Wiring: START -> classify -> (greeting | question) -> END.
# The conditional edge (not plain edges) is what makes classify branch.
def choose_next(state):
    return "greeting" if state.get("is_greeting") else "question"

graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", choose_next, {"greeting": "greeting", "question": "question"})
graph.add_edge("greeting", END)
graph.add_edge("question", END)

# A StateGraph must be compiled before it can be invoked.
app = graph.compile()
result = app.invoke({"text": "Hello there"})
print(result["response"])
```

This graph demonstrates branching behavior based on the input state. The full runnable
version is in `langgraph_toy_graph.py`.
