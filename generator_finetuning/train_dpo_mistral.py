import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig


def resolve_fulltext_dir() -> Path:
    """Resolve fulltext directory after repo refactors."""
    script_dir = Path(__file__).resolve().parent
    for root in [script_dir, *script_dir.parents]:
        candidate = root / "data" / "agora" / "fulltext"
        if candidate.exists():
            return candidate
    return script_dir / "fulltext"


def load_preferences(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Combined preferences file must be a JSON array")
    return data


def build_prompt(question: str, policy_name: Optional[str], context: str) -> str:
    """Build an instruction-style prompt for Mistral and include full document context."""
    return (
        "You are an expert policy analyst tasked with answering questions about AI policy and regulations. "
        "Provide direct, factual information grounded in the context. "
        "Cite relevant sources or document IDs where applicable, but do NOT use or generate external links. "
        "If the context does not contain enough information, state that explicitly instead of speculating. "
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Provide: a comprehensive answer with a direct answer to the question, and citations to relevant sources where necessary. "
        "Do not include URLs or external links in your answer. "
        "Answer: "
    )


def load_document_context(doc_id: Optional[int], fulltext_dir: Path) -> str:
    if doc_id is None:
        return "(No document_id provided.)"
    doc_path = fulltext_dir / f"{doc_id}.txt"
    if not doc_path.exists():
        return f"(Context file not found: {doc_path.name})"
    try:
        with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"(Error reading context: {e})"


def make_dpo_dataset(rows: List[Dict[str, Any]], fulltext_dir: Path) -> Dataset:
    prompts = []
    chosen = []
    rejected = []

    for r in rows:
        q = r.get("question", "")
        policy_name = r.get("policy_name")
        a1 = r.get("answer_1", "")
        a2 = r.get("answer_2", "")
        pref = r.get("preferred")
        doc_id = r.get("document_id")

        # Skip rows without preference
        if pref not in (1, 2):
            continue

        context = load_document_context(doc_id if isinstance(doc_id, int) else None, fulltext_dir)
        prompt = build_prompt(q, policy_name, context)

        if pref == 1:
            c, rj = a1, a2
        else:
            c, rj = a2, a1

        # Basic cleanup
        c = (c or "").strip()
        rj = (rj or "").strip()

        # Skip empty answers
        if not c or not rj:
            continue

        prompts.append(prompt)
        chosen.append(c)
        rejected.append(rj)

    if not prompts:
        raise ValueError("No valid preference rows found for DPO training")

    return Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})


def parse_args():
    p = argparse.ArgumentParser(description="Train Mistral-7B with DPO using preferences JSON")
    p.add_argument(
        "--prefs",
        type=str,
        default=str(Path(__file__).parent / "dpo_preferences_combined.json"),
        help="Path to combined preferences JSON",
    )
    p.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model to fine-tune",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "dpo_mistral_ckpt"),
        help="Directory to save checkpoints",
    )
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--max-length", type=int, default=2048, help="Max input length")
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    p.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    p.add_argument("--bf16", action="store_true", help="Use BF16 precision")
    return p.parse_args()


def main():
    args = parse_args()

    prefs_path = Path(args.prefs)
    rows = load_preferences(prefs_path)
    fulltext_dir = resolve_fulltext_dir()
    dset = make_dpo_dataset(rows, fulltext_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Ensure proper chat formatting tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None)),
        device_map="auto",
    )

    # Build DPO config (TRL expects DPOConfig, not HF TrainingArguments)
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        optim="paged_adamw_32bit",
        report_to=["none"],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL will create a reference copy unless you pass one
        args=training_args,
        train_dataset=dset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
