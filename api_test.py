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

def list_available_models():
    """Fetch and display available models from the API."""
    try:
        models = client.models.list()
        print("Available Models:")
        print("-" * 50)
        
        model_list = []
        for idx, model in enumerate(models.data, 1):
            print(f"{idx}. {model.id}")
            model_list.append(model.id)
        
        print("-" * 50)
        return model_list
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def select_model(model_list):
    """Allow user to select a model from the list."""
    while True:
        try:
            choice = input(f"\nSelect a model (1-{len(model_list)}) or enter model name directly: ").strip()
            
            # Check if input is a number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(model_list):
                    return model_list[idx]
                else:
                    print(f"Please enter a number between 1 and {len(model_list)}")
            else:
                # User entered model name directly
                return choice
        except KeyboardInterrupt:
            print("\nSelection cancelled.")
            return None
        except Exception as e:
            print(f"Invalid input: {e}")

def run_chat_completion(model_name, prompt, system_message="You are a helpful assistant.", 
                       temperature=0.7, max_tokens=10000):
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

def get_usage_info():
    """Fetch usage information for the API key."""
    usage_url = f"{BASE_URL}/usage/{API_KEY}"
    try:
        r = requests.get(usage_url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Failed to fetch usage: {e}")
        return None

def main():
    """Main function to orchestrate the model selection and chat completion."""
    print("=" * 50)
    print("OpenAI-Compatible Model Selector")
    print("=" * 50)
    
    # List available models
    model_list = list_available_models()
    
    if not model_list:
        print("No models available. Exiting.")
        return
    
    # Select a model
    selected_model = select_model(model_list)
    
    if not selected_model:
        print("No model selected. Exiting.")
        return
    
    print(f"\nSelected model: {selected_model}")
    
    # Get user prompt
    print("\nEnter your prompt (or press Enter for default):")
    user_prompt = input().strip()
    
    if not user_prompt:
        user_prompt = "Write a book in 2000 words."
        print(f"Using default prompt: {user_prompt}")
    
    # Run chat completion
    print(f"\nGenerating response with {selected_model}...")
    print("-" * 50)
    
    result = run_chat_completion(selected_model, user_prompt)
    
    if result:
        print("\nChatCompletion result:")
        print(result)
        print()
    
    # Display usage information
    print("-" * 50)
    usage_info = get_usage_info()
    if usage_info:
        print("\nUsage Information:")
        print(usage_info)

if __name__ == "__main__":
    main()