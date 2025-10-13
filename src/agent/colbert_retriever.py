import pickle
from ragatouille import RAGPretrainedModel


def get_content_map(path="./chunk_content/map.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

# Load the saved index (doesn’t need original RAG object)
def get_context(query: str, k=5, index_path="./.ragatouille/colbert/indexes/agora_index"):
    RAG = RAGPretrainedModel.from_index(index_path)

    results = RAG.search(query, k=k)
    content_map = get_content_map()

    # retrieve full chunk content
    for r in results:
        r["content"] = content_map[r["document_id"]]
    # fields: content, score, rank, document_id, passage_id, document_metadata
    return results

#print(get_context("What AI regulations exist in the One Big Beautiful Bill Act?"))