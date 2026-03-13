import pandas as pd
import json
import pickle
from ragatouille import RAGPretrainedModel

first_question = 1500
num_questions = 100
chunks_per_question = 50
question_file = "evaluation/questions_only.csv"
output_file = "evaluation/eval_questions_with_chunks.jsonl"

questions = pd.read_csv(question_file)
RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/agora_index")

def retrieve_docs(query):
    results = RAG.search(query, k=chunks_per_question)
    return results

with open("chunk_content/map.pkl", "rb") as f:
    chunk_content = pickle.load(f)

for idx, row in questions.iterrows():
    if idx < first_question: continue
    if idx > first_question + num_questions: break
    retrieved = retrieve_docs(row["question"])
    chunks = []
    for i, c in enumerate(retrieved):
        chunk = c["document_id"] + "\n"
        chunk += chunk_content[c["document_id"]]
        chunks.append(chunk)
    with open(output_file, "a") as f:
        f.write(json.dumps({
            "question": row["question"],
            "chunks": chunks
        }))
        f.write("\n")
    if (idx + 1) % 20 == 0:
        print(f"chunks for {idx + 1 - first_question} questions retrieved and saved")
