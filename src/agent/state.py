from typing import TypedDict, List
from langchain.schema import Document

class AgentState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    history: List[dict]