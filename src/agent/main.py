import os
from agent.graph import build_agent_graph
from utils import get_secrets

if __name__ == "__main__":
    os.environ["GEMINI_API_KEY"] = get_secrets()["GEMINI_API_KEY"]
    graph = build_agent_graph()
    question = ""
    result = graph.run({"question": question})
    print(result["answer"])