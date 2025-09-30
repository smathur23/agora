def basic_question_prompt(context, question, history):
    return f"""You are a helpful assistant specialized in AI regulation.
Conversation so far: 
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
