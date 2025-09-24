import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(provider: str = "gemini", params: dict = {}):
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=os.getenv("GEMINI_API_KEY"),
            **params
        )
    else:
        return None