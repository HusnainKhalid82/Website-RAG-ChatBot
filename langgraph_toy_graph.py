from langgraph.graph import END, START, StateGraph

# Toy LangGraph example: classify the input and route to greeting or question
# handling. This is the Day-1 warm-up that proves I understand StateGraph,
# nodes, edges, and conditional edges before building the real RAG graph.

graph = StateGraph(dict)


def classify(state):
    text = state.get("text", "").strip().lower()
    state["is_greeting"] = any(greet in text for greet in ["hi", "hello", "hey"])
    return state


def respond_greeting(state):
    state["response"] = "Hello! How can I help you today?"
    return state


def respond_question(state):
    state["response"] = "That sounds like a question. I would answer it here."
    return state


def choose_next(state):
    # Conditional edge: pick the next node based on state set by `classify`.
    return "greeting" if state.get("is_greeting") else "question"


graph.add_node("classify", classify)
graph.add_node("greeting", respond_greeting)
graph.add_node("question", respond_question)

# Wiring: START -> classify -> (greeting | question) -> END
graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", choose_next, {"greeting": "greeting", "question": "question"})
graph.add_edge("greeting", END)
graph.add_edge("question", END)

# A StateGraph must be compiled before it can be invoked.
app = graph.compile()


if __name__ == "__main__":
    for sample in ["Hello, how are you?", "What is a vector store?"]:
        result = app.invoke({"text": sample})
        print(f"input : {sample}")
        print(f"output: {result['response']}\n")
