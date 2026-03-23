import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig


def resolve_fulltext_dir(script_dir: Path, input_jsonl: Path) -> Path:
    """Resolve fulltext directory in repo data layout with backward-compatible fallback."""
    for root in [script_dir, *script_dir.parents]:
        candidate = root / "data" / "agora" / "fulltext"
        if candidate.exists():
            return candidate
    fallback = input_jsonl.parent / "fulltext"
    if fallback.exists():
        return fallback
    return script_dir / "fulltext"


def build_prompt(question: str, policy_name: Optional[str], context: str) -> str:
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


def load_questions_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if all(k in obj for k in ("document_id", "question")):
                    items.append(obj)
            except json.JSONDecodeError:
                continue
    return items


def load_full_document(document_id: Any, fulltext_dir: Path) -> str:
    try:
        doc_path = fulltext_dir / f"{document_id}.txt"
        if not doc_path.exists():
            return ""
        with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def build_model_and_tokenizer(model_id: str = "./dpo_mistral_merged", load_in_8bit: bool = True, int8_cpu_offload: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()

    if load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=int8_cpu_offload,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",  # allow offload when VRAM is tight
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
        )
    else:
        dtype = torch.float16 if use_cuda else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if use_cuda else None,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    return tokenizer, model


def make_pipeline(tokenizer, model, temperature: float = 0.2, top_p: float = 0.95, top_k: int = 40, max_new_tokens: int = 512):
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
        top_p=top_p,
        top_k=top_k,
    )


def answer_questions(
    input_jsonl: Path,
    output_json: Path,
    model_id: str = "./dpo_mistral_merged",
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    limit: Optional[int] = None,
    max_context_chars: Optional[int] = None,
    load_in_8bit: bool = True,
    int8_cpu_offload: bool = True,
):
    items = load_questions_jsonl(input_jsonl)
    if limit:
        items = items[:limit]

    print(f"Loaded {len(items)} questions from {input_jsonl}")

    tokenizer, model = build_model_and_tokenizer(model_id=model_id, load_in_8bit=load_in_8bit, int8_cpu_offload=int8_cpu_offload)
    gen = make_pipeline(tokenizer, model, temperature=temperature, max_new_tokens=max_new_tokens)

    script_dir = Path(__file__).resolve().parent
    fulltext_dir = resolve_fulltext_dir(script_dir, input_jsonl)
    results: List[Dict[str, Any]] = []

    for i, item in enumerate(items, 1):
        question = (item.get("question") or "").strip()
        if not question:
            continue
        policy_name = item.get("policy_name")
        doc_id = item.get("document_id")

        context_text = load_full_document(doc_id, fulltext_dir)
        if max_context_chars and max_context_chars > 0:
            context_text = context_text[:max_context_chars]

        prompt = build_prompt(question, policy_name, context_text)

        try:
            out = gen(prompt)[0]["generated_text"].strip()
        except Exception as e:
            print(f"Error generating for idx {i} (doc {doc_id}): {e}")
            continue

        results.append(
            {
                "document_id": doc_id,
                "policy_name": policy_name,
                "question": question,
                "answer": out,
                "temperature": temperature,
                "model_id": model_id,
            }
        )

        if i % 10 == 0:
            print(f"Processed {i}/{len(items)}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} answers to {output_json}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Answer questions with Mistral-7B-Instruct using full document context")
    parser.add_argument("--input", type=str, default=str(Path(__file__).parent / "generated_questions_any.jsonl"), help="Input JSONL file of questions")
    parser.add_argument("--output", type=str, default=str(Path(__file__).parent / "answers_any_mistral.json"), help="Output JSON file")
    parser.add_argument("--model", type=str, default=str(Path(__file__).parent / "dpo_mistral_merged"), help="Model directory or HF model ID")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens for generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--max-context-chars", type=int, default=None, help="Optionally truncate document context")
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit quantization (use fp16/fp32)")
    parser.add_argument("--no-int8-cpu-offload", action="store_true", help="Disable FP32 CPU offload for 8-bit modules")

    args = parser.parse_args()
    answer_questions(
        input_jsonl=Path(args.input),
        output_json=Path(args.output),
        model_id=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        max_context_chars=args.max_context_chars,
        load_in_8bit=(not args.no_8bit),
        int8_cpu_offload=(not args.no_int8_cpu_offload),
    )
