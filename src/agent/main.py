import os
from src.agent.graph import build_terminal_agent_graph
from src.utils import get_secrets

if __name__ == "__main__":
    #os.environ["GEMINI_API_KEY"] = get_secrets()["GEMINI_API_KEY"]
    agent = build_terminal_agent_graph()
    result = agent.invoke({})