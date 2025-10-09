import pickle
from colbert import Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig

def load_metadata(metadata_path):
    with open(metadata_path, "rb") as f:
        return pickle.load(f)

def load_id_map(id_map_path):
    with open(id_map_path, "rb") as f:
        return pickle.load(f)

def build_retriever(index_name="agora_index",
                    model_name="colbert-ir/colbertv2.0",
                    output_dir="colbert_indexes",
                    metadata_path="colbert_indexes/metadata.pkl",
                    id_map_path="colbert_indexes/docid_to_chunkid.pkl"):

    # Load metadata mapping (id → chunk dict)
    metadata_map = load_metadata(metadata_path)
    id_map = load_id_map(id_map_path)

    # Create a ColBERT searcher
    with Run().context(RunConfig(experiment="agora")):
        config = ColBERTConfig(root=output_dir)
        searcher = Searcher(index=index_name, checkpoint=model_name, config=config)

    def retrieve(query: str, k: int = 5):
        # Search the index
        doc_ids, ranks, scores = searcher.search(query, k=k)
        hits = []
        for doc_id, score in zip(doc_ids, scores):
            chunk_id = id_map[doc_id]
            chunk = metadata_map.get(chunk_id, {"text": "", "metadata": {}})
            hits.append({
                "id": chunk_id,
                "score": score,
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {})
            })
        return hits

    return retrieve