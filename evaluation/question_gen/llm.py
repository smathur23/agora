from openai import OpenAI
import requests

# Server configuration
BASE_URL = "http://aiforge.cs.purdue.edu:8001/v1"
API_KEY = "REMOVED"

# Initialize OpenAI client
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

def run_chat_completion(model_name, prompt, system_message="You are a helpful assistant.", 
                       temperature=1.0, max_tokens=500):
    """Run a chat completion with the selected model."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during chat completion: {e}")
        return None