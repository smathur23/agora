import math
import pickle
from ragatouille import RAGPretrainedModel
from typing import List, Dict, Tuple

import os
import pandas as pd
import torch


def load_csv_data(segments_path: str, documents_path: str):
    print("Loading CSV data...")
    segments_df = pd.read_csv(segments_path)
    documents_df = pd.read_csv(documents_path)
        
    print(f"Loaded {len(segments_df)} segments and {len(documents_df)} documents")

    return segments_df, documents_df

def create_document_chunks(segments_df, documents_df) -> List[Dict]:
    chunks = []

    for idx, row in segments_df.iterrows():
        source_doc_df = documents_df.loc[documents_df["AGORA ID"] == row["Document ID"]]

        source_doc = source_doc_df.iloc[0] if not source_doc_df.empty else None # Assuming AGORA_ID is a key for the documents table
        chunk = {
            'id': f"segment_{row['Document ID']}_{row['Segment position']}",
            'type': 'segment',
            'document_id': row['Document ID'],
            'segment_position': row['Segment position'],
            'official_name': str(source_doc['Official name']) if source_doc is not None else "",
            "casual_name": str(source_doc['Casual name']) if source_doc is not None and pd.notna(source_doc['Casual name']) else "",
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


def get_metadata_from_chunk(chunk: dict) -> dict:
    out = {
        "id": chunk["id"],
        "type": chunk["type"]
    }
    relevant_keys = ["link","agora_id","document_id","segment_position"]
    relevant_metadata = ['collections','most_recent_activity','most_recent_activity_date','segment_annotated','segment_validated']
    for key in relevant_keys:
        if key in chunk and chunk[key] != "":
            out[key] = chunk[key]
    for key in relevant_metadata:
        if key in chunk["metadata"] and chunk["metadata"][key] != "":
            out[key] = chunk["metadata"][key]
    return out

def get_relevant_data_from_chunk(chunk: dict) -> str:
    out = f"id: {chunk["id"]},\nofficial_name: {chunk["official_name"]},\ntext: {chunk["text"]}"
    relevant_keys = ["casual_name",'short_summary',"summary",'authority','tags',]
    relevant_metadata = ['non_operative','not_ai_related','proposed_date','primarily_government','primarily_private']
    for key in relevant_keys:
        if key in chunk and chunk[key] != "":
            out += f",\n{key}: {chunk[key]}"
    for key in relevant_metadata:
        if key in chunk["metadata"] and chunk["metadata"][key] != "":
            out += f",\n{key}: {chunk["metadata"][key]}"
    return out


def generate_colbert_embeddings(
        chunks,
        model_name: str = "colbert-ir/colbertv2.0",
        index_name: str = "agora_index",
    ):
    """
    Index documents with ColBERT instead of sentence-transformers.
    Returns the path to the ColBERT index.
    """

    texts = [get_relevant_data_from_chunk(chunk) for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [get_metadata_from_chunk(chunk) for chunk in chunks]

    for t in texts:
        if "id: " not in t:
            print("id not in text")
            print(t)
        if "casual_name" not in t:
            print("casual_name not in text")
            print(t) 

    RAG = RAGPretrainedModel.from_pretrained(model_name)

    # Clean metadata to avoid errors from inf or NaN values
    for i, m in enumerate(metadatas):
        clean = {}
        for k, v in m.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                #print(m)
                clean[k] = None
            else:
                clean[k] = v
        metadatas[i] = clean

    RAG.index(
        collection=texts,
        document_ids=ids,
        index_name=index_name,
        document_metadatas=metadatas
    )
    
    content_map = {chunks[i]['id']: texts[i] for i in range(len(chunks))}
    map_path = os.path.join("chunk_content", "map.pkl")
    os.makedirs("chunk_content", exist_ok=True)
    with open(map_path, "wb") as f:
        pickle.dump(content_map, f)

    return index_name, map_path

def create_colbert_index():
    segments_df, documents_df = load_csv_data("./data/agora/segments.csv", "./data/agora/documents.csv")
    print("Chunking data...")
    chunks = create_document_chunks(segments_df, documents_df)
    print("Creating ColBERT index...")
    index_name, map_path = generate_colbert_embeddings(chunks)
    print(f"Done. Index {index_name} created. Full chunk texts are stored at {map_path}")