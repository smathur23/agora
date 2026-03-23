import os
import json
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
import google.generativeai as genai
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def resolve_data_dir(script_dir: Path) -> Path:
    """Resolve AGORA data directory after repo refactors."""
    for root in [script_dir, *script_dir.parents]:
        candidate = root / "data" / "agora"
        if candidate.exists():
            return candidate
    return script_dir

class QuestionGenerator:
    """Generate training questions from policy documents for DPO fine-tuning."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Initialize the question generator with LLM."""
        api_key = os.getenv("GEMINI_KEY")
        if not api_key:
            raise ValueError("GEMINI_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = resolve_data_dir(self.base_dir)
        self.prompt_template = self._load_prompt_template()
        self.generation_count = 0  # Track number of generations
        
    def _load_prompt_template(self) -> str:
        """Load the question generation prompt template."""
        prompt_path = self.base_dir / "prompt_compl.txt"
        with open(prompt_path, 'r') as f:
            return f.read()
    
    def _load_document(self, doc_id: int) -> str:
        """Load document text from fulltext folder."""
        doc_path = self.data_dir / "fulltext" / f"{doc_id}.txt"
        if not doc_path.exists():
            doc_path = self.base_dir / "fulltext" / f"{doc_id}.txt"
        if not doc_path.exists():
            raise FileNotFoundError(f"Document {doc_id} not found at {doc_path}")
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load document metadata from CSV."""
        csv_path = self.data_dir / "documents.csv"
        if not csv_path.exists():
            csv_path = self.base_dir / "documents.csv"
        return pd.read_csv(csv_path)
    
    def _count_words(self, text: str) -> int:
        """Count the number of words in the text."""
        return len(text.split())
    
    def _is_valid_length(self, text: str, min_words: int = 300, max_words: int = 1200) -> bool:
        """Check if document is within the acceptable word count range."""
        word_count = self._count_words(text)
        return min_words <= word_count <= max_words
    
    def _truncate_text(self, text: str, max_chars: int = 15000) -> str:
        """Truncate text to fit within token limits while preserving complete sentences."""
        if len(text) <= max_chars:
            return text
        
        truncated = text[:max_chars]
        # Try to cut at last complete sentence
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.8:  # Only if we don't lose too much
            return truncated[:last_period + 1]
        return truncated
    
    def _check_rate_limit(self):
        """Check if we need to wait to avoid rate limits."""
        self.generation_count += 1
        if self.generation_count % 15 == 0:
            print(f"\nReached {self.generation_count} generations. Waiting 60 seconds to avoid rate limits...")
            for remaining in range(60, 0, -1):
                print(f"\rResuming in {remaining} seconds...  ", end='', flush=True)
                time.sleep(1)
            print("\nResuming processing...\n")

    def generate_questions(self, doc_id: int, policy_text: str, policy_name: str) -> List[Dict]:
        """Generate questions for a single document using the LLM."""
        # Truncate if needed
        truncated_text = self._truncate_text(policy_text)
        
        # Fill in the prompt template with all placeholders
        prompt = self.prompt_template.replace("{{policy_text}}", truncated_text)
        prompt = prompt.replace("{{document_id}}", str(doc_id))
        prompt = prompt.replace("{{policy_name}}", policy_name)
        
        try:
            # Check rate limit before generation
            self._check_rate_limit()
            
            # Generate questions
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            questions_data = json.loads(response_text)
            
            # Validate format
            if not isinstance(questions_data, list):
                print(f"Warning: Doc {doc_id} returned non-list response")
                return []
            
            # Validate each question object
            validated_questions = []
            for i, item in enumerate(questions_data):
                if not self._validate_question_format(item, doc_id):
                    print(f"Warning: Doc {doc_id} question {i+1} has invalid format, skipping")
                    continue
                validated_questions.append(item)
            
            return validated_questions
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for doc {doc_id}: {e}")
            print(f"Response: {response.text[:300]}...")
            return []
        except Exception as e:
            print(f"Error generating questions for doc {doc_id}: {e}")
            return []
    
    def _validate_question_format(self, item: Dict, expected_doc_id: int) -> bool:
        """Validate that a question item has the correct format."""
        required_fields = ["document_id", "policy_name", "question", "category"]
        
        # Check all required fields exist
        if not all(field in item for field in required_fields):
            missing = [f for f in required_fields if f not in item]
            print(f"  Missing fields: {missing}")
            return False
        
        # Validate field types
        if not isinstance(item["document_id"], int):
            print(f"  document_id must be int, got {type(item['document_id'])}")
            return False
        
        if not isinstance(item["policy_name"], str) or not item["policy_name"].strip():
            print(f"  policy_name must be non-empty string")
            return False
        
        if not isinstance(item["question"], str) or not item["question"].strip():
            print(f"  question must be non-empty string")
            return False
        
        if not isinstance(item["category"], str) or not item["category"].strip():
            print(f"  category must be non-empty string")
            return False
        
        # Validate document_id matches
        if item["document_id"] != expected_doc_id:
            print(f"  document_id mismatch: expected {expected_doc_id}, got {item['document_id']}")
            return False
        
        # Validate category value
        if item["category"] != "compliance":
            print(f"  category must be 'compliance', got '{item['category']}'")
            return False
        
        return True

    def process_all_documents(self, output_file: str = "generated_questions.jsonl", 
                            max_docs: Optional[int] = None,
                            min_words: int = 300,
                            max_words: int = 1200,
                            sample_seed: Optional[int] = None) -> None:
        """Process all documents and generate questions.
        
        Args:
            output_file: Output file path for generated questions
            max_docs: Maximum number of documents to process (None for all). If provided, a random sample of documents is selected.
            min_words: Minimum word count for documents to process
            max_words: Maximum word count for documents to process
            sample_seed: Optional random seed for reproducible sampling of documents
        """
        # Load metadata
        metadata_df = self._load_metadata()
        
        # Get all document IDs
        doc_ids = []
        fulltext_dir = self.data_dir / "fulltext"
        if not fulltext_dir.exists():
            fulltext_dir = self.base_dir / "fulltext"
        for file in fulltext_dir.glob("*.txt"):
            doc_ids.append(int(file.stem))
        
        doc_ids = sorted(doc_ids)
        
        print(f"Found {len(doc_ids)} total documents")
        print(f"Filtering for documents between {min_words} and {max_words} words...")
        
        # Filter documents by word count
        valid_doc_ids = []
        skipped_too_short = 0
        skipped_too_long = 0
        
        for doc_id in tqdm(doc_ids, desc="Filtering documents"):
            try:
                policy_text = self._load_document(doc_id)
                word_count = self._count_words(policy_text)
                
                if word_count < min_words:
                    skipped_too_short += 1
                elif word_count > max_words:
                    skipped_too_long += 1
                else:
                    valid_doc_ids.append(doc_id)
                    
            except FileNotFoundError:
                continue
        
        print(f"\nFiltering results:")
        print(f"  Valid documents: {len(valid_doc_ids)}")
        print(f"  Skipped (too short): {skipped_too_short}")
        print(f"  Skipped (too long): {skipped_too_long}")
        
        if max_docs:
            if len(valid_doc_ids) > max_docs:
                rng = random.Random(sample_seed)
                sampled_doc_ids = rng.sample(valid_doc_ids, k=max_docs)
                valid_doc_ids = sampled_doc_ids
                seed_msg = f" with seed {sample_seed}" if sample_seed is not None else ""
                print(f"  Randomly sampled {max_docs} out of {len(valid_doc_ids)} valid documents{seed_msg}")
            else:
                print(f"  Requested max_docs={max_docs}, but only {len(valid_doc_ids)} valid documents available")
        
        print(f"\nProcessing {len(valid_doc_ids)} documents...")
        print(f"Rate limiting: Will pause for 60 seconds after every 15 generations\n")
        
        results = []
        output_path = self.base_dir / output_file
        
        # Clear the output file if it exists
        if output_path.exists():
            output_path.unlink()
        
        # Process each document
        for doc_id in tqdm(valid_doc_ids, desc="Generating questions"):
            try:
                # Load document
                policy_text = self._load_document(doc_id)
                word_count = self._count_words(policy_text)
                
                # Get metadata using the correct column name
                doc_meta = metadata_df[metadata_df['AGORA ID'] == doc_id]
                
                if doc_meta.empty:
                    print(f"\nWarning: No metadata found for doc {doc_id}")
                    policy_name = f"Document {doc_id}"
                else:
                    # Use 'Casual name' if available, otherwise 'Official name'
                    if pd.notna(doc_meta.iloc[0]['Casual name']) and doc_meta.iloc[0]['Casual name'].strip():
                        policy_name = doc_meta.iloc[0]['Casual name']
                    elif pd.notna(doc_meta.iloc[0]['Official name']) and doc_meta.iloc[0]['Official name'].strip():
                        policy_name = doc_meta.iloc[0]['Official name']
                    else:
                        policy_name = f"Document {doc_id}"
                
                # Generate questions (now returns fully formatted objects)
                question_objects = self.generate_questions(doc_id, policy_text, policy_name)
                
                if not question_objects:
                    print(f"\nWarning: No valid questions generated for doc {doc_id} ({word_count} words)")
                    continue
                
                # Store and write results
                for question_obj in question_objects:
                    # Add word count metadata
                    question_obj["word_count"] = word_count
                    results.append(question_obj)
                    
                    # Write incrementally to avoid losing progress
                    with open(output_path, 'a') as f:
                        f.write(json.dumps(question_obj) + '\n')
                
            except FileNotFoundError:
                print(f"\nSkipping doc {doc_id}: file not found")
                continue
            except Exception as e:
                print(f"\nError processing doc {doc_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nGenerated {len(results)} questions from {len(valid_doc_ids)} documents")
        print(f"Total API calls made: {self.generation_count}")
        print(f"Results saved to {output_path}")
        
        # Also save as regular JSON for convenience
        json_output = output_path.with_suffix('.json')
        with open(json_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Also saved to {json_output}")



def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate questions for DPO fine-tuning")
    parser.add_argument("--max-docs", type=int, help="Maximum number of documents to process")
    parser.add_argument("--min-words", type=int, default=300, 
                       help="Minimum word count for documents (default: 300)")
    parser.add_argument("--max-words", type=int, default=1200,
                       help="Maximum word count for documents (default: 1200)")
    parser.add_argument("--output", type=str, default="generated_questions.jsonl", 
                       help="Output file for questions")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite",
                       help="Model to use for generation")
    parser.add_argument("--sample-seed", type=int, default=None,
                       help="Optional random seed for reproducible sampling when --max-docs is set")
    
    args = parser.parse_args()
    
    # Generate questions
    generator = QuestionGenerator(model_name=args.model)
    generator.process_all_documents(
        output_file=args.output, 
        max_docs=args.max_docs,
        min_words=args.min_words,
        max_words=args.max_words,
        sample_seed=args.sample_seed
    )

if __name__ == "__main__":
    main()