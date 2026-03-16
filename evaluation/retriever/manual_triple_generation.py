online = True # set to false if you have a file with already retrieved chunks for questions

import pandas as pd
import json
import pickle
if online:
    from ragatouille import RAGPretrainedModel

if online:
    questions = pd.read_csv("retriever_finetuning/questions_processed.csv")
    RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/agora_index")
else:
    source_file = "evaluation/retriever/eval_questions_with_chunks.jsonl"

def retrieve_docs(query):
    results = RAG.search(query, k=50)
    return results

with open("evaluation/retriever/discarded_questions_online.txt" if online else "evaluation/retriever/discarded_questions_offline.txt", "r") as f:
    num_discarded = sum(1 for line in f)

with open("evaluation/retriever/manually_labeled_questions_online.jsonl" if online else "evaluation/retriever/manually_labeled_questions_offline.jsonl", "r") as f:
    num_processed = sum(1 for line in f)

if online:
    with open("chunk_content/map.pkl", "rb") as f:
        chunk_content = pickle.load(f)

start_index = input("Enter index of question to start with or ENTER for next unprocessed question ")
if start_index == "":
    start_index = num_discarded + num_processed
else:
    start_index = int(start_index)

if online:
    for idx, row in questions.iterrows():
        if idx < start_index:
            continue
        q = input(f"""

    STARTING QUESTION {idx}:
    {row["question"]}
    ENTER to continue, n to discard question

    """)
        if q != "":
            with open("evaluation/retriever/discarded_questions_online.txt", "a") as f:
                f.write(f"{idx}\n")
            continue
        
        retrieved = retrieve_docs(row["question"])
        relevant = []
        irrelevant = []
        for i, c in enumerate(retrieved):
            print("\n\nQuestion: ")
            print(row["question"])
            print("\n")
            print(chunk_content[c["document_id"]])
            print(f"\nChunk {i + 1}/50")
            resp = input("'ENTER' if chunk is relevant, 'n' if chunk is not relevant: ")
            if resp == "":
                relevant.append(c["document_id"])
            else:
                irrelevant.append(c["document_id"])
        with open("evaluation/retriever/manually_labeled_questions_online.jsonl", "a") as f:
            f.write(json.dumps({"question": row["question"], "positive_ids": relevant, "negative_ids": irrelevant}))
            f.write("\n")
else:
    with open(source_file, "r") as f:
        i = -1
        for line in f:
            i += 1
            row = json.loads(line)
            q = input(f"""

STARTING QUESTION {i}:
{row["question"]}
ENTER to continue, 'n' to discard question

""")
            if q != "":
                with open("evaluation/retriever/discarded_questions_offline.txt", "a") as f:
                    f.write(f"{i}\n")
                continue
            retrieved = row["chunks"]
            relevant = []
            irrelevant = []
            for i, c in enumerate(retrieved):
                print("\n\nQuestion: ")
                print(row["question"])
                print("\n")
                print(c)
                print(f"\nChunk {i + 1}/20")
                resp = input("'ENTER' if chunk is relevant, 'n' if chunk is not relevant: ")
                if resp == "":
                    relevant.append(c.split("\n")[0])
                else:
                    irrelevant.append(c.split("\n")[0])
            with open("evaluation/retriever/manually_labeled_questions_offline.jsonl", "a") as f:
                f.write(json.dumps({"question": row["question"], "positive_ids": relevant, "negative_ids": irrelevant}))
                f.write("\n")