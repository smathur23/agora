import pandas as pd
from evaluation.prompts import system_prompt
from evaluation.llm import run_chat_completion

prompts_df = pd.read_csv("evaluation/prompts.csv")
prompts = prompts_df["prompt"]
relevant_ids = prompts_df["relevant_ids"]

n = len(prompts)

# ========================================================= #
# Generate questions with llm                               #
# ========================================================= #
print("prompts generated, starting question generation")
model_name = "gemma3:27b"

system_prompt = system_prompt()
questions = []
for i, p in enumerate(prompts):
    question = run_chat_completion(model_name, p, system_message=system_prompt)
    #print(question)
    questions.append(question)
    if (i + 1) % 20 == 0:
        print(f"{i + 1}/{n}")
        print(len(questions))
        # Save every 20 questions
        out = pd.DataFrame({
            "prompt": prompts[:len(questions)],
            "question": questions,
            "relevant_ids": relevant_ids[:len(questions)] 
        })
        out.to_csv("./evaluation/questions.csv", index=False)
