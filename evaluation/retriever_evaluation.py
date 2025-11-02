from ragatouille import RAGPretrainedModel
import pandas as pd
from src.agent.prompts import basic_question_prompt

RAG = RAGPretrainedModel.from_index("./.ragatouille/colbert/indexes/agora_index")

def mrr(relevant_docs, retrieved_docs):
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            return 1.0 / rank
    return 0.0

def recall(relevant_docs, retrieved_docs, k):
    retrieved_top_k = retrieved_docs[:k]
    hits = len(set(retrieved_top_k) & set(relevant_docs))
    total_relevant = len(relevant_docs)
    return hits / total_relevant if total_relevant > 0 else 0.0

def average_precision(relevant_docs,retrieved_docs, k):
    relevant_docs = set(relevant_docs)
    retrieved_docs = retrieved_docs[:min(len(retrieved_docs), k)]
    num_hits = 0
    score = 0.0
    for i, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            num_hits += 1
            score += num_hits / i
    return score / min(len(relevant_docs), k)

def retrieve_docs(query):
    results = RAG.search(query, k=20)
    out = []
    for r in results:
        out.append(r["document_id"])
    return out
    
if __name__ == "__main__":
    questions = pd.read_csv("./evaluation/questions_processed.csv")
    # naive prompting
    mrr_count = 0
    recall_5 = 0
    recall_10 = 0
    recall_20 = 0
    map_5 = 0
    map_10 = 0
    map_20 = 0
    n = len(questions)
    for index, row in questions.iterrows():
        retrieved = retrieve_docs(row["questions"])

        relevant = row["relevant_ids"].split(";")
        relevant = [i for i in relevant if "segment" in i]

        mrr_count += mrr(relevant, retrieved)
        recall_5 += recall(relevant, retrieved, 5)
        recall_10 += recall(relevant, retrieved, 10)
        recall_20 += recall(relevant, retrieved, 20)
        map_5 += average_precision(relevant, retrieved, 5)
        map_10 += average_precision(relevant, retrieved, 10)
        map_20 += average_precision(relevant, retrieved, 20)
    mrr_count /= n
    recall_5 /= n
    recall_10 /= n
    recall_20 /= n
    map_5 /= n
    map_10 /= n
    map_20 /= n
    print(f"n: {n}")
    print(f"mrr: {mrr_count}")
    print(f"recall@5: {recall_5}, recall@10: {recall_10}, recall@20: {recall_20}")
    print(f"MAP@5: {map_5}, MAP@10: {map_10}, MAP@20: {map_20}")

    


