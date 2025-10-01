import os
import pickle
import numpy as np
from typing import List, Dict, Tuple

def load_index(index_dir: str = "embeddings_output") -> Dict:
    """Load the saved search index from disk."""
    index_path = os.path.join(index_dir, "search_index.pkl")
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    return index

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b + 1e-10)

def search(query: str, index: Dict, model_name: str = "all-MiniLM-L6-v2", top_k: int = 5) -> List[Tuple[str, Dict, float]]:
    """Search the index for top_k most similar chunks to the query."""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    query_vec = model.encode([query])[0]

    embeddings = index["embeddings"]
    chunks = index["texts"]
    metadatas = index["metadatas"]


    # Compute similarities
    sims = [cosine_similarity(query_vec, emb) for emb in embeddings]

    # Sort by similarity
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)

    # Return top_k chunks
    results = []
    for idx, score in ranked[:top_k]:
        results.append((
            chunks[idx],
            metadatas[idx],
            score
        ))
    
    return results
