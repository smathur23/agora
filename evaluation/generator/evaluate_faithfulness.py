from datasets import Dataset
import json
from pathlib import Path
from typing import Any
from ragas.metrics import Faithfulness
from ragas import evaluate
from ragas.llms import llm_factory
from openai import OpenAI


def resolve_data_paths() -> tuple[Path, Path]:
    """Resolve answers file and fulltext directory across repo layouts."""
    script_dir = Path(__file__).resolve().parent
    for root in [script_dir, *script_dir.parents]:
        data_fulltext = root / "data" / "agora" / "fulltext"
        if data_fulltext.exists():
            answers_path = script_dir / "answers_any_mistral_normal.json"
            return answers_path, data_fulltext
    return script_dir / "answers_any_mistral_normal.json", script_dir / "fulltext"

# Load answers
ANSWERS_PATH, FULLTEXT_DIR = resolve_data_paths()
with open(ANSWERS_PATH, "r") as f:
    data = json.load(f)

def load_full_document(document_id: Any, fulltext_dir: Path) -> str:
    path = fulltext_dir / f"{document_id}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")

def attach_context(item):
    return {
        "question": item["question"],
        "answer": item["answer"],
        "contexts": [
            load_full_document(item["document_id"], FULLTEXT_DIR)
        ],
    }

dataset = Dataset.from_list([attach_context(x) for x in data])

# vLLM OpenAI-compatible client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

llm = llm_factory(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    client=client,
)

faithfulness = Faithfulness(llm=llm)

results = evaluate(
    dataset,
    metrics=[faithfulness],
)

print(results)
