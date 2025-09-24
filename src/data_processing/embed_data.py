import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import pickle
from datetime import datetime
import re

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    if not text or len(text) < chunk_size:
        return [text] if text else []
    
    # Split by sentences first to maintain context
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed chunk size
        if len(current_chunk) + len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                # Single sentence too long, split by words
                words = sentence.split()
                for i in range(0, len(words), chunk_size//10):
                    word_chunk = " ".join(words[i:i + chunk_size//10])
                    chunks.append(word_chunk)
                current_chunk = ""
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def assign_labels(metadata: Dict, chunk_index: int, total_chunks: int) -> Dict:
    """Assign labels and metadata to text chunks."""
    labels = {
        'chunk_index': chunk_index,
        'total_chunks': total_chunks,
        'document_id': metadata.get('document_id', ''),
        'country': metadata.get('country', ''),
        'policy_type': metadata.get('policy_type', ''),
        'year': metadata.get('year', ''),
        'sector': metadata.get('sector', ''),
        'summary': metadata.get('summary', ''),
        'created_at': datetime.now().isoformat()
    }
    
    # Add any additional metadata fields
    for key, value in metadata.items():
        if key not in labels:
            labels[f'meta_{key}'] = value
    
    return labels

def generate_embeddings(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Generate embeddings for text chunks using sentence transformers."""
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return np.array([])

def prepare_embedding_data(processed_data: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """Prepare text chunks and labels for embedding."""
    all_chunks = []
    all_labels = []
    
    for record in processed_data:
        text = record['text']
        metadata = record['metadata']
        filename = record['filename']
        
        # Add filename to metadata if not present
        if 'filename' not in metadata:
            metadata['filename'] = filename
        
        # Create chunks
        chunks = chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            labels = assign_labels(metadata, i, len(chunks))
            labels['source_file'] = record['file_path']
            labels['content_hash'] = record['content_hash']
            all_labels.append(labels)
    
    return all_chunks, all_labels

def save_embeddings(embeddings: np.ndarray, labels: List[Dict], chunks: List[str], 
                   output_dir: str = 'embeddings_output') -> str:
    """Save embeddings, labels, and chunks to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save embeddings as numpy array
    embeddings_path = os.path.join(output_dir, f'embeddings_{timestamp}.npy')
    np.save(embeddings_path, embeddings)
    
    # Save labels as JSON
    labels_path = os.path.join(output_dir, f'labels_{timestamp}.json')
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    
    # Save chunks as text file
    chunks_path = os.path.join(output_dir, f'chunks_{timestamp}.txt')
    with open(chunks_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"=== CHUNK {i} ===\n")
            f.write(chunk)
            f.write("\n\n")
    
    # Save metadata summary
    summary_path = os.path.join(output_dir, f'summary_{timestamp}.json')
    summary = {
        'total_chunks': len(chunks),
        'embedding_dimension': embeddings.shape[1] if embeddings.size > 0 else 0,
        'unique_documents': len(set(label.get('source_file', '') for label in labels)),
        'countries': list(set(label.get('country', '') for label in labels if label.get('country'))),
        'policy_types': list(set(label.get('policy_type', '') for label in labels if label.get('policy_type'))),
        'created_at': datetime.now().isoformat()
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Embeddings saved to: {output_dir}")
    print(f"Summary: {summary}")
    
    return output_dir

def create_embeddings_index(embeddings: np.ndarray, labels: List[Dict], chunks: List[str]) -> Dict:
    """Create an index for fast similarity search."""
    index = {
        'embeddings': embeddings,
        'labels': labels,
        'chunks': chunks,
        'dimension': embeddings.shape[1] if embeddings.size > 0 else 0,
        'size': len(labels)
    }
    
    # Create reverse lookup dictionaries
    country_index = {}
    policy_type_index = {}
    
    for i, label in enumerate(labels):
        country = label.get('country', '')
        policy_type = label.get('policy_type', '')
        
        if country:
            if country not in country_index:
                country_index[country] = []
            country_index[country].append(i)
        
        if policy_type:
            if policy_type not in policy_type_index:
                policy_type_index[policy_type] = []
            policy_type_index[policy_type].append(i)
    
    index['country_index'] = country_index
    index['policy_type_index'] = policy_type_index
    
    return index

def process_and_embed_data(processed_data: List[Dict], model_name: str = 'all-MiniLM-L6-v2',
                          output_dir: str = 'embeddings_output') -> str:
    """Main function to process data and create embeddings."""
    print("Preparing text chunks and labels...")
    chunks, labels = prepare_embedding_data(processed_data)
    
    if not chunks:
        print("No text chunks to process!")
        return ""
    
    print(f"Generated {len(chunks)} text chunks from {len(processed_data)} documents")
    
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks, model_name)
    
    if embeddings.size == 0:
        print("Failed to generate embeddings!")
        return ""
    
    print("Saving embeddings and metadata...")
    output_path = save_embeddings(embeddings, labels, chunks, output_dir)
    
    print("Creating search index...")
    index = create_embeddings_index(embeddings, labels, chunks)
    
    # Save index for agent use
    index_path = os.path.join(output_dir, 'search_index.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump(index, f)
    
    print(f"Processing complete! Output saved to: {output_path}")
    return output_path
