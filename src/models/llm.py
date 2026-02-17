import os
import torch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from dotenv import load_dotenv

def get_llm(model_id: str = "mistralai/Mistral-7B-Instruct-v0.3", provider: str = "hf", params: dict = {}):
    load_dotenv()
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_id, 
            google_api_key=os.getenv("GEMINI_KEY"),
            **params
        )
    elif provider == "hf":
        # Optional: allow caller to override default generation params
        max_new_tokens = params.pop("max_new_tokens", 1024)
        temperature = params.pop("temperature", 0.2)

        # Use AutoModelForCausalLM for better quantization support
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Configure 8-bit quantization via bitsandbytes
        int8_cpu_offload = params.pop("int8_cpu_offload", True)
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=int8_cpu_offload,
        )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=quant_config,
            )
        except ValueError as e:
            # Helpful fallback message for low VRAM or offload configuration
            raise ValueError(
                f"Failed to load {model_id} in 8-bit. "
                f"Try passing params={'{'}'int8_cpu_offload': True{'}'} or reducing max_new_tokens, "
                f"or switch to 4-bit quantization. Original error: {e}"
            )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return HuggingFacePipeline(pipeline=pipe)
    return None