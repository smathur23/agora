from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(provider: str = "gemini", params: dict = None):
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", **params)
    else:
        return None