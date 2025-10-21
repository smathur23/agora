
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

def retrieve_docs(query):
    results = RAG.search(query, k=20)
    out = []
    for r in results:
        out.append(r["document_id"])
    return out
    
if __name__ == "__main__":
    questions = pd.read_csv("./evaluation/test_questions.csv")
    # naive prompting
    mrr_count = 0
    recall_5 = 0
    recall_10 = 0
    n = len(questions)
    for index, row in questions.iterrows():
        retrieved = retrieve_docs(row["question"])

        relevant = row["relevant_documents"].split(",")

        mrr_count += mrr(relevant, retrieved)
        recall_5 += recall(relevant, retrieved, 5)
        recall_10 += recall(relevant, retrieved, 10)
    mrr_count /= n
    recall_5 /= n
    recall_10 /= n
    print(f"n: {n}")
    print(f"mrr: {mrr_count}, recall@5: {recall_5}, recall@10: {recall_10}")

    


