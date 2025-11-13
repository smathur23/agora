import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split

input_csv = "evaluation/questions_with_negatives2.csv"
test_jsonl = "evaluation/test.jsonl"

# Helper to split id lists like "id1;id2"
def split_ids(x):
    if type(x) == list: return x
    return x.split(";")

df = pd.read_csv(input_csv)

# remove document ids and other misformed values
for idx, row in df.iterrows():
    relevant = row["relevant_ids"].split(";")
    relevant = [i for i in relevant if "segment_" in i and len(i) <= 16]
    row["relevant_ids"] = ";".join(relevant)
    true = row["true_negatives"].split(";")
    true = [i for i in true if "segment_" in i and len(i) <= 16]
    row["true_negatives"] = ";".join(true)
    useful = row["useful_negatives"].split(";")
    useful = [i for i in true if "segment_" in i and len(i) <= 16]
    row["useful_negatives"] = ";".join(useful)
    random = row["random_negatives"].split(";")
    random = [i for i in random if "segment_" in i and len(i) <= 16]
    row["random_negatives"] = random
    combo = row["combo_negatives"].split(";")
    combo = [i for i in combo if "segment_" in i and len(i) <= 16]
    row["combo_negatives"] = combo
df.to_csv("evaluation/cleaned_with_negatives.csv")

# 80/20 split
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)



def df_to_jsonl(df, filepath, negative_col=None):
    with open(filepath, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            query = row["question"]
            positives = split_ids(row["relevant_ids"])
            if negative_col is None:
                record = {
                    "query": query,
                    "positive_document_ids": positives,
                }
                f.write(json.dumps(record) + "\n")
            else:
                negatives = split_ids(row[negative_col])
                record = {
                    "query": query,
                    "positive_document_ids": positives,
                    "negative_document_ids": negatives
                }
                f.write(json.dumps(record) + "\n")


# write outputs
df_to_jsonl(train_df, "evaluation/train_true.jsonl", "true_negatives")
df_to_jsonl(train_df, "evaluation/train_close.jsonl", "useful_negatives")
df_to_jsonl(train_df, "evaluation/train_combo.jsonl", "combo_negatives")
df_to_jsonl(train_df, "evaluation/train_naive.jsonl", "random_negatives")
df_to_jsonl(test_df, test_jsonl)
