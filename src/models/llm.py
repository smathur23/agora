import os
import torch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, MistralForCausalLM, AutoTokenizer

def get_llm(provider: str = "hf", model_id: str = "mistralai/Mistral-7B-Instruct-v0.3", params: dict = {}):
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_id, 
            google_api_key=os.getenv("GEMINI_API_KEY"),
            **params
        )
    elif provider == "hf":
        ModelForCausalLM = MistralForCausalLM if "mistral" in model_id else AutoModelForCausalLM    
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ModelForCausalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto"
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.2
        )
    return HuggingFacePipeline(pipeline=pipe)