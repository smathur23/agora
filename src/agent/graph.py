from langgraph.graph import StateGraph, END
from langchain.schema import Document
from src.agent.state import AgentState
from src.agent.prompts import basic_question_prompt, format_for_instruct
from src.models.llm import get_llm
from src.agent.retriever import load_index, search
from src.agent.colbert_retriever import build_retriever

def build_agent_graph():
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = get_llm(model_id)

    graph = StateGraph(AgentState)

    def retrieve(state):
        """ Get relevant context from data given question """
        index = load_index()
        results = search(state["question"], index)
        docs = [
            Document(
                page_content=chunk_text,
                metadata={**metadata, "score": float(score)}
            )
            for chunk_text, metadata, score in results
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
        
        # Extract content from response
        answer_text = result.content if hasattr(result, 'content') else str(result)
        
        updated_history = history + [{"user": question, "assistant": answer_text}]

        return {
            "question": question,
            "context": docs,
            "answer": answer_text,
            "history": updated_history
        }

    graph.add_node("retriever", retrieve)
    graph.add_node("answer", answer)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "answer")
    graph.add_edge("answer", END)

    agent = graph.compile()

    return agent

def build_terminal_agent_graph():
    """Terminal version with interactive input loop"""
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = get_llm(model_id)

    graph = StateGraph(AgentState)

    def user_input(state):
        """ Get question from user """
        question = input("You: ")
        return {**state, "question": question}

    def retrieve(state):
        """ Get relevant context from data given question """
        retriever = build_retriever()

        results = retriever(state["question"])
    
        return {
            "question": state["question"],
            "context": results
        }
    
    def answer(state):
        """ Get llm answer to question given history and context """
        question = state["question"]
        docs = state["context"]
        history = state.get("history", [])

        context = "\n\n".join([str(d) for d in docs])
        history_text = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history])

        prompt = basic_question_prompt(context, state["question"], history_text)
        instruct_prompt = format_for_instruct(prompt, model_id)
        result = llm.invoke(instruct_prompt)
        
        # Extract content from response
        answer_text = result.content if hasattr(result, 'content') else str(result)
        print("AI Assistant: \n" + answer_text)
        
        updated_history = history + [{"user": question, "assistant": answer_text}]

        return {
            "question": question,
            "context": docs,
            "answer": answer_text,
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