from langgraph.graph import StateGraph, END
from langchain.schema import Document
from src.agent.state import AgentState
from src.agent.prompts import basic_question_prompt
from src.models.llm import get_llm
from src.agent.retriever import load_index, search

def build_agent_graph():
    llm = get_llm()

    graph = StateGraph(AgentState)

    def user_input(state):
        """ Get question from user """
        question = input("You: ")
        return {**state, "question": question}

    def retrieve(state):
        """ Get relevant context from data given question """
        index = load_index("embeddings_output")
        results = search(state["question"], index)
        docs = [
            Document(
                page_content=text,
                metadata={**metadata, "score": float(score)}
            )
            for text, metadata, score in results
        ]
    
        return {
            "question": state["question"],
            "context": docs
        }
    
    def answer(state):
        """ Get llm answer to question given history and context """
        question = state["question"]
        docs = state["context"]
        history = state.get("history", [])

        context = "\n\n".join([d.page_content for d in docs])
        history_text = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history])

        prompt = basic_question_prompt(context, state["question"], history_text)
        result = llm.invoke(prompt)
        print("AI Assistant: \n" + result)
        updated_history = history + [{"user": question, "assistant": result}]

        return {
            "question": question,
            "context": docs,
            "answer": result,
            "history": updated_history
        }
    
    def should_continue(state: AgentState) -> str:
        """ Decide whether to loop back for another question or end """
        if state["question"].lower() in {"exit", "quit"}:
            return END
        return "retriever"

    graph.add_node("user_input", user_input)
    graph.add_node("retriever", retrieve)
    graph.add_node("answer", answer)

    graph.set_entry_point("user_input")

    graph.add_edge("retriever", "answer")
    graph.add_edge("answer", "user_input")
    graph.add_conditional_edges("user_input", should_continue)

    agent = graph.compile()

    return agent