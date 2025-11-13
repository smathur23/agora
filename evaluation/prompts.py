def system_prompt():
    return """"You are an AI policy expert creating questions to use to train and evaluate a retriever in a RAG pipeline.
    You will be given a description of the type of question to ask, and possibly the text and title of one or more AI policy documents that the RAG system should retrieve to answer the question.
    You may be provided with tags, that are used by researchers to categorize policy documents, and authorities, which are governments or organizations that create policy.
    Be creative. The questions you generate should be worded in different ways.
    Respond each time with one single question."""

def trends_over_tags(tag=None, year=None, authority=None):
    prompt = "Ask one single question about AI policy trends "
    if authority is not None: prompt += f"on policy created by authority: {authority} "
    if tag is not None: prompt += f"relating to tag: {tag} "
    if year is not None: prompt += f"since {year}.\n"
    else: prompt += f"over the course of time.\n"
    return prompt
    
def status_of_tags(tag, authority):
    prompt = f"Ask a question about the status of AI policy relating to tag: {tag} created by the authority of: {authority}.\n"
    return prompt

def specific_doc(doc_title, doc_text, name_provided=False):
    prompt = f"""Ask a question that could be answered by the content of the following policy document.
    doc name: {doc_title}

    (start of document)
    
    {doc_text}
    
    (end of document)
    
    Do{" not" if not name_provided else ""} provide the name of the document in the question.\n"""
    return prompt

def compare_docs(doc1_title, doc1_text, doc2_title, doc2_text, name_provided=False):
    prompt = f"""Ask a qustion that could be answered by analysis of the following 2 policy documents.
    document 1 name: {doc1_title}
    
    (start of document 1)

    {doc1_text}
    
    (end of document 1)
    
    document 2 name: {doc2_title}

    (start of document 2)
    
    {doc2_text}
    
    (end of document 2)
    
    Do{" not" if not name_provided else ""} provide the names of the documents in the question.\n"""
    return prompt

def open_ended_policy_questions():
    pass 

def comparison_of_authorities(auth_1, auth_2, tag=None):
    prompt = "Ask a question about the differences in AI policy "
    if tag is not None: prompt += f"relating to tag: {tag} "
    prompt += f"between policy created by authorities {auth_1} and {auth_2}.\n"
    return prompt

