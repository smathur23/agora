from langgraph.graph import StateGraph
from langchain.schema import Document
from src.agent.state import AgentState
from src.agent.prompts import basic_question_prompt
from src.models.llm import get_llm
from src.agent.retriever import load_index, search

def build_agent_graph():
    llm = get_llm()

    graph = StateGraph(AgentState)

    def retrieve(state):
        index = load_index("embeddings_output")
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
        docs = state["context"]
        context = "\n\n".join([d.page_content for d in docs])
        prompt = basic_question_prompt(context, state["question"])
        print(prompt)
        return {"answer": llm.invoke(prompt)}

    graph.add_node("retriever", retrieve)
    graph.add_node("answer", answer)
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "answer")

    agent = graph.compile()

    return agent