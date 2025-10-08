from transformers import AutoTokenizer

def basic_question_prompt(context, question, history):
    return f"""Conversation so far: 
{history}

Based on the following context from the AGORA dataset, please answer the question. Provide specific references and citations where possible. Don't include any unwanted links in the references.

Context:
{context}

Question: {question}

Please provide a comprehensive answer with:
1. Direct answer to the question
2. Relevant citations and references
3. Document IDs and sources where information was found

Answer:
"""

def format_for_instruct(prompt: str, model_id: str):  
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in AI regulation."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
