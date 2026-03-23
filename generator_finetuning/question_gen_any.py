import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv


CATEGORY_FILES = {
    "compl": "prompt_compl.txt",
    "def": "prompt_def.txt",
    "eval": "prompt_eval.txt",
    "impl": "prompt_impl.txt",
    "stake": "prompt_stake.txt",
    "sum_exp": "prompt_sum_exp.txt",
}


def resolve_data_dir(script_dir: Path) -> Path:
    """Resolve AGORA data directory after repo refactors."""
    for root in [script_dir, *script_dir.parents]:
        candidate = root / "data" / "agora"
        if candidate.exists():
            return candidate
    return script_dir


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class QuestionGeneratorAny:
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", categories: Optional[List[str]] = None):
        load_dotenv()
        api_key = os.getenv("GEMINI_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_KEY env var not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.base_dir = Path(__file__).parent
        self.data_dir = resolve_data_dir(self.base_dir)
        self.categories = categories or list(CATEGORY_FILES.keys())
        self.templates = self._load_templates(self.categories)
        self.generation_count = 0

    def _load_templates(self, categories: List[str]) -> Dict[str, str]:
        templates = {}
        for c in categories:
            fname = CATEGORY_FILES.get(c)
            if not fname:
                continue
            templates[c] = load_text(self.base_dir / fname)
        if not templates:
            raise ValueError("No templates loaded; check categories.")
        return templates

    def _load_document(self, doc_id: int) -> str:
        p = self.data_dir / "fulltext" / f"{doc_id}.txt"
        if not p.exists():
            p = self.base_dir / "fulltext" / f"{doc_id}.txt"
        if not p.exists():
            raise FileNotFoundError(f"Fulltext not found for document_id={doc_id}")
        return load_text(p)

    def _load_metadata(self) -> pd.DataFrame:
        metadata_path = self.data_dir / "documents.csv"
        if not metadata_path.exists():
            metadata_path = self.base_dir / "documents.csv"
        return pd.read_csv(metadata_path)

    def _truncate_text(self, text: str, max_chars: int = 15000) -> str:
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.8:
            return truncated[:last_period+1]
        return truncated

    def _check_rate_limit(self):
        self.generation_count += 1
        if self.generation_count % 15 == 0:
            import time
            time.sleep(120)

    def _validate_item(self, item: Dict, expected_doc_id: int) -> bool:
        required = ["document_id", "policy_name", "question", "category"]
        if not all(k in item for k in required):
            return False
        if item["document_id"] != expected_doc_id:
            return False
        return True

    def generate_questions_for_doc(self, doc_id: int, policy_name: str) -> List[Dict]:
        text = self._load_document(doc_id)
        truncated = self._truncate_text(text)
        # pick a random category template each time
        cat = random.choice(self.categories)
        tmpl = self.templates[cat]
        prompt = tmpl.replace("{{policy_text}}", truncated).replace("{{document_id}}", str(doc_id)).replace("{{policy_name}}", policy_name)
        try:
            self._check_rate_limit()
            resp = self.model.generate_content(prompt)
            # Prefer structured access; fallback to .text
            content = None
            try:
                if resp and getattr(resp, "candidates", None):
                    for cand in resp.candidates:
                        if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                            for part in cand.content.parts:
                                if hasattr(part, "text") and part.text:
                                    content = part.text
                                    break
                        if content:
                            break
                if not content and hasattr(resp, "text"):
                    content = resp.text
            except Exception:
                content = getattr(resp, "text", None)

            if not content:
                raise ValueError("Empty response content from model")

            # Sanitize common code-fence wrappers like ```json ... ``` or ``json
            cleaned = content.strip()
            # Remove leading triple backticks with optional language
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
                # drop language spec if present (e.g., json)
                cleaned = cleaned.lstrip().split("\n", 1)
                cleaned = cleaned[1] if len(cleaned) > 1 else cleaned[0]
            # Remove trailing triple backticks
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            # Remove single backtick fenced language markers like ``json
            if cleaned.lower().startswith("``json"):
                cleaned = cleaned.split("\n", 1)
                cleaned = cleaned[1] if len(cleaned) > 1 else cleaned[0]

            # Finally parse JSON
            items = json.loads(cleaned)
            # keep only valid items
            valid = [it for it in items if self._validate_item(it, doc_id)]
            return valid
        except Exception as e:
            print(f"Error generating questions for doc_id={doc_id}: {e}")
            return []

    def process_all_documents(self, output_file: str = "generated_questions_any.jsonl", max_docs: Optional[int] = None, min_words: int = 300, max_words: int = 1200, sample_seed: Optional[int] = None) -> None:
        # collect document ids from fulltext
        fulltext_dir = self.data_dir / "fulltext"
        if not fulltext_dir.exists():
            fulltext_dir = self.base_dir / "fulltext"
        doc_ids = sorted([int(p.stem) for p in fulltext_dir.glob("*.txt") if p.stem.isdigit()])

        # filter by length
        valid_doc_ids = []
        for doc_id in tqdm(doc_ids, desc="Filtering documents"):
            try:
                text = self._load_document(doc_id)
            except Exception:
                continue
            wc = len(text.split())
            if min_words <= wc <= max_words:
                valid_doc_ids.append(doc_id)

        # sample subset if requested
        if max_docs:
            if sample_seed is not None:
                random.seed(sample_seed)
            valid_doc_ids = random.sample(valid_doc_ids, min(max_docs, len(valid_doc_ids)))
        print(valid_doc_ids)

        out_path = self.base_dir / output_file
        if out_path.exists():
            out_path.unlink()

        metadata = self._load_metadata()
        name_map = {int(row.document_id): row.policy_name for _, row in metadata.iterrows() if "document_id" in metadata.columns and "policy_name" in metadata.columns}

        results = []
        for doc_id in tqdm(valid_doc_ids, desc="Generating questions"):
            policy_name = name_map.get(doc_id, f"Document {doc_id}")
            items = self.generate_questions_for_doc(doc_id, policy_name)
            results.extend(items)
            # append to jsonl
            with open(out_path, "a", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")

        print(f"Generated {len(results)} questions across {len(valid_doc_ids)} documents")


def main():
    parser = argparse.ArgumentParser(description="Generate questions from any category (compl, def, eval, impl, stake, sum_exp)")
    parser.add_argument("--max-docs", type=int, help="Maximum number of documents to process")
    parser.add_argument("--min-words", type=int, default=300, help="Minimum word count")
    parser.add_argument("--max-words", type=int, default=1200, help="Maximum word count")
    parser.add_argument("--output", type=str, default="generated_questions_any.jsonl", help="Output JSONL file")
    parser.add_argument("--model", type=str, default="gemma-3-27b-it", help="Model name")
    parser.add_argument("--categories", type=str, nargs='*', default=None, help="Subset of categories to sample from")
    parser.add_argument("--sample-seed", type=int, default=None, help="Optional random seed")

    args = parser.parse_args()
    gen = QuestionGeneratorAny(model_name=args.model, categories=args.categories)
    gen.process_all_documents(output_file=args.output, max_docs=args.max_docs, min_words=args.min_words, max_words=args.max_words, sample_seed=args.sample_seed)


if __name__ == "__main__":
    main()
