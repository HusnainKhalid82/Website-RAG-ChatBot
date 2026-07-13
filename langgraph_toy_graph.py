from langgraph.graph import StateGraph

# Toy LangGraph example: classify input and route to greeting or question handling.

graph = StateGraph(dict)


def classify(state):
    text = state.get("text", "").strip().lower()
    if any(greet in text for greet in ["hi", "hello", "hey"]):
        state["is_greeting"] = True
    else:
        state["is_greeting"] = False
    return state


def respond_greeting(state):
    state["response"] = "Hello! How can I help you today?"
    return state


def respond_question(state):
    state["response"] = "That sounds like a question. I would answer it here."
    return state


def choose_next(state):
    return "greeting" if state.get("is_greeting") else "question"


graph.add_node("classify", classify)
graph.add_node("greeting", respond_greeting)
graph.add_node("question", respond_question)
graph.add_conditional_edges("classify", choose_next)


if __name__ == "__main__":
    result = graph.invoke({"text": "Hello, how are you?"})
    print(result["response"])
