import json
import pandas as pd

segments = pd.read_csv("data/agora/segments.csv")

def get_random_negative(positives):
    while True:
        row = segments.sample(n=1)
        if f"segment_{row['AGORA ID']}_{row['Segment position']}" not in positives:
            return f"segment_{row['AGORA ID']}_{row['Segment position']}"

triples = []

with open("retriever_finetuning/manually_labeled_questions.jsonl", "r") as f:
    for line in f:
        record = json.loads(line)
        if len(record["positive_ids"]) == 0:
            continue
        negs = record["negative_ids"]
        if len(negs) == 0:
            negs = [get_random_negative(record["postive_ids"])]
        for p in record["positive_ids"]:
            for n in negs:
                triples.append({
                    "query": record["question"],
                    "positive_example": p,
                    "negative_example": n
                })
with open("retriever_finetuning/train.jsonl", "w") as f:
    for t in triples:
        json.dump(t, f)
        f.write("\n")