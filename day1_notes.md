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
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 1) Load a single web page
loader = WebBaseLoader("https://example.com")
docs = loader.load()

# 2) Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 3) Embed the chunks
embeddings = OpenAIEmbeddings(openai_api_key="YOUR_OPENAI_KEY")
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
from langgraph.graph import StateGraph

# Create a simple graph
graph = StateGraph(dict)

# 1. classify the input

def classify(state):
    text = state.get("text", "").strip().lower()
    if any(greet in text for greet in ["hi", "hello", "hey"]):
        state["is_greeting"] = True
    else:
        state["is_greeting"] = False
    return state

# 2. respond to greeting

def respond_greeting(state):
    state["response"] = "Hello! How can I help you today?"
    return state

# 3. respond to question

def respond_question(state):
    state["response"] = "That sounds like a question. I would answer it here."
    return state

# Add nodes and edges
graph.add_node("classify", classify)
graph.add_node("greeting", respond_greeting)
graph.add_node("question", respond_question)
graph.add_edge("classify", "greeting")
graph.add_edge("classify", "question")

def choose_next(state):
    return "greeting" if state.get("is_greeting") else "question"

graph.add_conditional_edges("classify", choose_next)

# Run the graph
result = graph.invoke({"text": "Hello there"})
print(result["response"])
```

This graph demonstrates branching behavior based on the input state.
