import math
import pickle
from ragatouille import RAGPretrainedModel
from typing import List, Dict, Tuple
import os
import pandas as pd
import torch

MAX_DOC_LEN = -1

def load_csv_data(segments_path: str, documents_path: str):
    print("Loading CSV data...")
    segments_df = pd.read_csv(segments_path)
    documents_df = pd.read_csv(documents_path)
        
    print(f"Loaded {len(segments_df)} segments and {len(documents_df)} documents")

    return segments_df, documents_df

def create_fixed_size_chunks(segments_df, documents_df, size, overlap):
    chunks = []
    for idx, row in documents_df.iterrows():
        id = row["AGORA ID"]
        count = 1
        doc_data = {
            'official_name': str(row['Official name']),
            'casual_name': str(row['Casual name']) if pd.notna(row['Casual name']) else "",
            'collections': row.get('Collections', ''),
            'most_recent_activity': row.get('Most recent activity', ''),
            'most_recent_activity_date': row.get('Most recent activity date', ''),
            'proposed_date': row.get('Proposed date', ''),
            'primarily_government': row.get('Primarily applies to the government', False),
            'primarily_private': row.get('Primarily applies to the private sector', False),
            'authority': str(row['Authority']) if pd.notna(row['Authority']) else "",
            "tags": ""
        }
        doc_metadata = {
            'agora_id': id,
            'link': str(row['Link to document']) if pd.notna(row['Link to document']) else "",
        }
        curr_chunk = {
            "id": f"chunk_{id}_{count}",
            "text": "",
            "data": {k: doc_data[k] for k in doc_data},
            "metadata": {k: doc_metadata[k] for k in doc_metadata},
        }
        curr_chunk["metadata"]["segments"] = []
        doc_segments = segments_df[segments_df["Document ID"] == id]
        for idx, row in doc_segments.iterrows():
            tags = str(row['Tags']) if pd.notna(row['Tags']) else ""
            #ai_related = row["not_ai_related"]
            #non_operative = row["non_operative"]
            text = row["Text"].split(" ")
            print(len(text))
            return
            curr_chunk["data"]["tags"] += tags if len(curr_chunk["data"]["tags"]) == 0 else ";" + tags
            #curr_chunk["data"]["not_ai_related"] = ai_related
            #curr_chunk["non_operative"] = non_operative

            remaining_text_needed = size - len(curr_chunk["text"])
            while remaining_text_needed < len(text):
                curr_chunk["text"] += text[:remaining_text_needed]
                text = text[remaining_text_needed:]
                chunks.append(curr_chunk)
                count += 1

                curr_chunk = {
                    "id": f"chunk_{id}_{count}",
                    "text": "",
                    "data": {k: doc_data[k] for k in doc_data},
                    "metadata": {k: doc_metadata[k] for k in doc_metadata}
                }
                curr_chunk["data"]["tags"] = tags
                remaining_text_needed = size
            curr_chunk["text"] += text
        print(chunks)
        return

def create_document_chunks(segments_df, documents_df) -> List[Dict]:
    chunks = []
    for idx, row in documents_df.iterrows():
        id = row["AGORA ID"]
        if not os.path.exists(f"data/agora/fulltext/{id}.txt"):
            if pd.notna(row['Long summary']): text = row["Long summary"]
            elif pd.notna(row['Short summary']): text = row["Short summary"]
            else: text = ""
        else:
            with open(f"data/agora/fulltext/{id}.txt", "r") as f:
                text = f.read()
        if text == "":
            print(f"No text for document {id} could be found.")
        if len(text) > MAX_DOC_LEN:
            chunks += create_segments_from_doc(segments_df, row)
        else:
            chunk = create_chunk_from_row(row, text)
            chunks.append(chunk)
    return chunks

def create_chunk_from_row(row, text):
    return {
        'id': f"document_{row['AGORA ID']}",
        "text": text,
        "data": {
            'type': 'document',
            'official_name': str(row['Official name']),
            'casual_name': str(row['Casual name']) if pd.notna(row['Casual name']) else "",
            'tags': str(row['Tags']) if pd.notna(row['Tags']) else "",
            'collections': row.get('Collections', ''),
            'most_recent_activity': row.get('Most recent activity', ''),
            'most_recent_activity_date': row.get('Most recent activity date', ''),
            'proposed_date': row.get('Proposed date', ''),
            'primarily_government': row.get('Primarily applies to the government', False),
            'primarily_private': row.get('Primarily applies to the private sector', False),
            'authority': str(row['Authority']) if pd.notna(row['Authority']) else "",
        },
        "metadata": {
            'agora_id': row['AGORA ID'],
            'link': str(row['Link to document']) if pd.notna(row['Link to document']) else "",
        }
        
    }

def create_segments_from_doc(segments_df, doc):
    segments = segments_df[segments_df["Document ID"] == doc["AGORA ID"]]
    chunks = []
    for idx, row in segments.iterrows():
        chunk = create_chunk_from_row(doc, row["Text"])
        chunk["id"] = f"segment_{row['Document ID']}_{row['Segment position']}"
        chunk["data"]["non_operative"] = row.get('Non-operative', False)
        chunk["data"]["not_ai_related"] = row.get("Not AI-related", False)
        chunk["data"]["tags"] = str(row['Tags']) if pd.notna(row['Tags']) else ""
        chunk["data"]["type"] = "segment"
        chunk["metadata"]["segment_position"] = row["Segment position"]
        chunk["metadata"]["segment_annotated"] = row.get('Segment annotated', False)
        chunk["metadata"]["segment_validated"] = row.get('Segment validated', False)

        chunks.append(chunk)

    return chunks

def format_chunk_data(data):
    out = ""
    for field in data:
        if data[field] == None or data[field] == "":
            continue
        field_value = str(data[field]).replace("\n", " ")
        out += field + ": " + field_value + "\n"
    out += "\n\n"
    if "authority" not in out:
        print(out)
    return out

def generate_colbert_embeddings(
        chunks,
        model_name: str = ".ragatouille/colbert/none/2025-12/02/labeled_only/checkpoints/colbert",
        index_name: str = "labeled_only_index",
    ):
    """
    Index documents with ColBERT instead of sentence-transformers.
    Returns the path to the ColBERT index.
    """

    texts = [format_chunk_data(chunk["data"]) + chunk["text"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

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