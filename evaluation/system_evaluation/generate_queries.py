from evaluation.llm import run_chat_completion
from evaluation.system_evaluation.blog_queries import system_prompt, single_post, multi_post
import os

model_name = "gemma3:27b"

def single_post_questions(filepath):
    with open(filepath, "r") as f:
        text = f.read()
    title = filepath.split("/")[-1]
    prompt = single_post(title, text)
    queries = run_chat_completion(model_name, prompt, system_message=system_prompt)
    print(queries)

if __name__ == "__main__":
    base_dir = "evaluation/system_evaluation/blog_posts/"
    blog_posts = os.listdir(base_dir)
    for b in blog_posts:
        single_post_questions(base_dir + b)