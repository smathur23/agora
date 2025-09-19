from langgraph.graph import StateGraph
from src.models.llm import get_llm
from agent.retriever import get_retriever

def build_agent_graph():
    retriever = get_retriever()
    llm = get_llm()

    graph = StateGraph()

    def retrieve(state):
        return {"docs": retriever.get_relevant_documents(state["question"])}
    
    def answer(state):
        docs = state["docs"]
        context = "\n\n".join([d.page_content for d in docs])
        return {"answer": llm.invoke(f"Context:\n{context}\n\nQuestion: {state['question']}")}

    graph.add_node("retriever", retrieve)
    graph.add_node("answer", answer)
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "answer")

    return graph