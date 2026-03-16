from datasets import Dataset
import json
from pathlib import Path
from typing import Any
from ragas.metrics import Faithfulness
from ragas import evaluate
from ragas.llms import llm_factory
from openai import OpenAI

# Load answers
with open("answers_any_mistral_normal.json", "r") as f:
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
            load_full_document(item["document_id"], Path("fulltext"))
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
