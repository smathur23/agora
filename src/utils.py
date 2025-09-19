import json

def get_secrets():
    try:
        with open('../secrets.json', 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("Error: 'secrets.json' not found in root directory.")
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from 'secrets.json'.")