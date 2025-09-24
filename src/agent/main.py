import os
from src.agent.graph import build_agent_graph
from src.utils import get_secrets

if __name__ == "__main__":
    os.environ["GEMINI_API_KEY"] = get_secrets()["GEMINI_API_KEY"]
    agent = build_agent_graph()
    question = "When is the EU AI law passed?"
    result = agent.invoke({"question": question})
    print(result["answer"])