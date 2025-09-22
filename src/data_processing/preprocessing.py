import os
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional
import hashlib

def extract_text(file_path: str) -> str:
    """Extract text for txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error extracting TXT {file_path}: {e}")
        return ""

def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
    
    # Remove headers/footers patterns (common in policy docs)
    text = re.sub(r'Page \d+.*?\n', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Normalize case
    text = text.strip()
    
    return text

def read_csv_metadata(csv_path: str) -> pd.DataFrame:
    """Read and process CSV metadata files."""
    try:
        df = pd.read_csv(csv_path)
        
        # Clean text columns
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            df[col] = df[col].astype(str).apply(lambda x: clean_text(x) if x != 'nan' else '')
        
        return df
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return pd.DataFrame()

def generate_content_hash(text: str) -> str:
    """Generate hash for content deduplication."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def integrate_data(policy_texts: Dict[str, str], metadata_df: pd.DataFrame, 
                  id_column: str = 'document_id') -> List[Dict]:
    """Integrate extracted text with CSV metadata."""
    integrated_data = []
    
    for file_path, text in policy_texts.items():
        # Extract filename for matching
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        # Find matching metadata
        metadata_row = None
        if id_column in metadata_df.columns:
            matches = metadata_df[metadata_df[id_column].str.contains(base_name, na=False)]
            if not matches.empty:
                metadata_row = matches.iloc[0].to_dict()
        
        # Create integrated record
        record = {
            'file_path': file_path,
            'filename': filename,
            'text': text,
            'content_hash': generate_content_hash(text),
            'text_length': len(text),
            'metadata': metadata_row or {}
        }
        
        integrated_data.append(record)
    
    return integrated_data

def deduplicate_data(integrated_data: List[Dict]) -> List[Dict]:
    """Remove duplicate entries based on content hash."""
    seen_hashes = set()
    deduplicated = []
    
    for record in integrated_data:
        content_hash = record['content_hash']
        if content_hash not in seen_hashes and record['text'].strip():
            seen_hashes.add(content_hash)
            deduplicated.append(record)
    
    print(f"Removed {len(integrated_data) - len(deduplicated)} duplicate entries")
    return deduplicated

def process_policy_files(data_folder: str) -> List[Dict]:
    """Main function to process all policy files and integrate with metadata."""
    policy_texts = {}
    metadata_dfs = []
    
    # Process all files in data folder
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            file_path = os.path.join(root, file)
            
            if file.endswith(('.pdf', '.docx', '.txt')):
                print(f"Extracting text from: {file}")
                text = extract_text(file_path)
                if text:
                    cleaned_text = clean_text(text)
                    policy_texts[file_path] = cleaned_text
            
            elif file.endswith('.csv'):
                print(f"Reading metadata from: {file}")
                df = read_csv_metadata(file_path)
                if not df.empty:
                    metadata_dfs.append(df)
    
    # Combine all metadata
    combined_metadata = pd.concat(metadata_dfs, ignore_index=True) if metadata_dfs else pd.DataFrame()
    
    # Integrate data
    integrated_data = integrate_data(policy_texts, combined_metadata)
    
    # Deduplicate
    final_data = deduplicate_data(integrated_data)
    
    print(f"Processed {len(final_data)} unique policy documents")
    return final_data
