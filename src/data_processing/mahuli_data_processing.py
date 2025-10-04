import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import json
from typing import List, Dict, Tuple
import os
from datetime import datetime

def load_csv_data(segments_path: str, documents_path: str):
    print("Loading CSV data...")
    segments_df = pd.read_csv(segments_path)
    documents_df = pd.read_csv(documents_path)
        
    print(f"Loaded {len(segments_df)} segments and {len(documents_df)} documents")

    return segments_df, documents_df


def create_document_chunks(segments_df, documents_df) -> List[Dict]:
    chunks = []

    for idx, row in segments_df.iterrows():
        chunk = {
            'id': f"segment_{row['Document ID']}_{row['Segment position']}",
            'type': 'segment',
            'document_id': row['Document ID'],
            'segment_position': row['Segment position'],
            'text': str(row['Text']),
            'summary': str(row['Summary']) if pd.notna(row['Summary']) else "",
            'tags': str(row['Tags']) if pd.notna(row['Tags']) else "",
            'metadata': {
                'non_operative': row.get('Non-operative', False),
                'not_ai_related': row.get('Not AI-related', False),
                'segment_annotated': row.get('Segment annotated', False),
                'segment_validated': row.get('Segment validated', False)
            }
        }
        chunks.append(chunk)

    for idx, row in documents_df.iterrows():
        chunk = {
            'id': f"document_{row['AGORA ID']}",
            'type': 'document',
            'agora_id': row['AGORA ID'],
            'official_name': str(row['Official name']),
            'casual_name': str(row['Casual name']) if pd.notna(row['Casual name']) else "",
            'text': f"{row['Official name']} - {row['Short summary']} - {row['Long summary']}",
            'short_summary': str(row['Short summary']) if pd.notna(row['Short summary']) else "",
            'long_summary': str(row['Long summary']) if pd.notna(row['Long summary']) else "",
            'authority': str(row['Authority']) if pd.notna(row['Authority']) else "",
            'link': str(row['Link to document']) if pd.notna(row['Link to document']) else "",
            'tags': str(row['Tags']) if pd.notna(row['Tags']) else "",
            'metadata': {
                'collections': row.get('Collections', ''),
                'most_recent_activity': row.get('Most recent activity', ''),
                'most_recent_activity_date': row.get('Most recent activity date', ''),
                'proposed_date': row.get('Proposed date', ''),
                'primarily_government': row.get('Primarily applies to the government', False),
                'primarily_private': row.get('Primarily applies to the private sector', False)
            }
        }
        chunks.append(chunk)

    return chunks

def generate_embeddings(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Generate embeddings for text chunks using sentence transformers."""
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return np.array([])

def save_embeddings(embeddings: np.ndarray, output_dir: str = 'embeddings_output') -> str:
    """Save embeddings to file"""
    print("Saving embeddings")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save embeddings as numpy array
    embeddings_path = os.path.join(output_dir, f'embeddings_{timestamp}.npy')
    np.save(embeddings_path, embeddings)
    
    print(f"Embeddings saved to: {output_dir}")
    
    return output_dir

def create_embeddings_index(texts: List[str], ids: List[str], embeddings: np.ndarray, metadatas: List[Dict],
                            output_dir: str = 'embeddings_output') -> Dict:
    """Create an index for fast similarity search."""
    print("Creating index")
    index = {
        'embeddings': embeddings,
        'texts': texts,
        'ids': ids,
        'metadatas': metadatas
    }

    index_path = os.path.join(output_dir, 'search_index.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump(index, f)
    print("Index saved")
    
    return index

def create_vector_database():
    segments_df, documents_df = load_csv_data("./data/agora/segments.csv", "./data/agora/documents.csv")
    print("Creating document chunks...")
    chunks = create_document_chunks(segments_df, documents_df)

    print("Generating embeddings...")
    texts = [chunk['text'] for chunk in chunks]
    embeddings = generate_embeddings(texts)

    ids = [chunk['id'] for chunk in chunks]
    metadatas = [{'type': chunk['type'], **chunk['metadata']} for chunk in chunks]

    output_path = save_embeddings(embeddings)
    create_embeddings_index(texts, ids, embeddings, metadatas)

    print(f"Processing complete! Output saved to: {output_path}")