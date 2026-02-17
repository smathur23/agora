system_prompt = """You are an AI policy expert creating questions and answers to use to train and evaluate a RAG system.
You will be given the text from one or more blog posts written by AI policy researchers containing
information and analysis about AI policy. Use the given text as a factual basis for writing questions and creating answers.
Be creative. The questions you generate should be worded in different ways. The questions should be about AI policy, not the blog posts themselves.
Respond each time with 5 question/answer pairs unless told otherwise."""

def single_post(title, text):
    return f"""Given the following blog post, create 5 AI policy related questions that can be answered using the blog post, and provide their answers
blog post title: {title}

blog post text: 
{text}"""

def multi_post(titles, posts):
    prompt = "Given the following blog posts, create 5 AI policy related questions that can be answered by information contained in the posts, and provide their answers.\nAsk questions that require information from at least 2 of the posts to answer.\n"
    for i, t in enumerate(titles):
        prompt += f"""blog post {i} title: {t}

blog post {i} text: {text[i]}\n\n"""