import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import (
	AutoTokenizer,
	AutoModelForCausalLM,
	MistralForCausalLM,
	pipeline,
)


def load_questions(input_path: Path) -> List[Dict[str, Any]]:
	"""Load questions from a JSON file that contains a list of dicts.

	Expected keys per item: document_id, policy_name, question (others ignored).
	"""
	with open(input_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	# Filter only items that have required fields
	cleaned = []
	for item in data:
		if all(k in item for k in ("document_id", "question")):
			cleaned.append(item)
	return cleaned


def build_model_and_tokenizer(model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"):
	"""Load tokenizer and model without quantization.

	- If CUDA is available: use float16 with device_map="auto" (no 4-bit/8-bit quantization)
	- Else: CPU float32 fallback
	"""
	tokenizer = AutoTokenizer.from_pretrained(model_id)

	use_cuda = torch.cuda.is_available()

	ModelForCausalLM = MistralForCausalLM if "mistral" in model_id.lower() else AutoModelForCausalLM

	if use_cuda:
		# GPU, no quantization; use FP16
		model = ModelForCausalLM.from_pretrained(
			model_id,
			device_map="auto",
			torch_dtype=torch.float16,
			low_cpu_mem_usage=True,
		)
	else:
		# CPU fallback (float32)
		model = ModelForCausalLM.from_pretrained(
			model_id,
			device_map=None,
			torch_dtype=torch.float32,
			low_cpu_mem_usage=True,
		)

	# Ensure pad token is set
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	return tokenizer, model


def make_pipeline(tokenizer, model, temperature: float, max_new_tokens: int = 512):
	"""Create a text-generation pipeline with specified temperature."""
	return pipeline(
		task="text-generation",
		model=model,
		tokenizer=tokenizer,
		do_sample=True,
		temperature=temperature,
		max_new_tokens=max_new_tokens,
		pad_token_id=tokenizer.eos_token_id,
		return_full_text=False,
	)


def build_prompt(question: str, policy_name: Optional[str], context: str) -> str:
	"""Build an instruction-style prompt for Mistral and include full document context."""
	policy_line = f"Policy: {policy_name}\n" if policy_name else ""
	# Use explicit instruction formatting compatible with instruct models
	# return (
	# 	f"[INST] You are an expert assistant answering questions about public policy and regulations.\n"
	# 	f"Answer clearly and concisely, grounding your answer ONLY in the provided context.\n"
	# 	f"If the context is insufficient to answer, state that explicitly without speculating.\n"
	# 	f"{policy_line}"
	# 	f"Context (verbatim document text):\n{context}\n\n"
	# 	f"Question: {question}\n\n"
	# 	f"Provide: (1) a direct answer grounded in the context, (2) brief reasoning. [/INST]"
	# )
	return (
		"You are an expert assistant answering questions about public policy and regulations."
        "Provide direct, factual information grounded in the context."
        "Cite relevant sources or document IDs where applicable."
        "If the context does not contain enough information, state that explicitly instead of speculating."
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Provide: a comprehensive answer with a direct answer to the question, and citations to relevant sources where necessary."
	)


def load_full_document(document_id: Any, fulltext_dir: Path) -> str:
	"""Load the full text for a document_id from the fulltext directory.

	Expects files named like "<document_id>.txt".
	Returns an empty string if the file doesn't exist.
	"""
	try:
		doc_path = fulltext_dir / f"{document_id}.txt"
		if not doc_path.exists():
			print(f"Warning: fulltext not found for document_id={document_id} at {doc_path}")
			return ""
		with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
			return f.read()
	except Exception as e:
		print(f"Error reading fulltext for document_id={document_id}: {e}")
		return ""


def generate_two_answers(
	pipe_cold, pipe_hot, prompt: str, seed_cold: int = 42, seed_hot: int = 1234
) -> Tuple[str, str]:
	"""Generate two answers with different temperatures using two pipelines."""
	# Set seeds for reproducibility (best-effort; sampling still stochastic)
	torch.manual_seed(seed_cold)
	out1 = pipe_cold(prompt)[0]["generated_text"].strip()

	torch.manual_seed(seed_hot)
	out2 = pipe_hot(prompt)[0]["generated_text"].strip()

	return out1, out2


def process_questions(
	input_json: Path,
	output_json: Path,
	model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
	temp_cold: float = 0.4,
	temp_hot: float = 0.7,
	max_new_tokens: int = 512,
	limit: Optional[int] = None,
	max_context_chars: Optional[int] = None,
) -> Dict[str, Any]:
	"""Main processing: load questions, generate two answers each, write JSON output."""
	items = load_questions(input_json)
	if limit:
		items = items[:limit]

	print(f"Loaded {len(items)} questions from {input_json}")

	tokenizer, model = build_model_and_tokenizer(model_id=model_id)
	print("Model loaded.")

	pipe_cold = make_pipeline(tokenizer, model, temperature=temp_cold, max_new_tokens=max_new_tokens)
	pipe_hot = make_pipeline(tokenizer, model, temperature=temp_hot, max_new_tokens=max_new_tokens)
	print(f"Pipelines ready (temps: cold={temp_cold}, hot={temp_hot}).")

	results: List[Dict[str, Any]] = []
	start_time = time.time()

	# Resolve fulltext directory (same folder as this script)/fulltext
	fulltext_dir = Path(__file__).parent / "fulltext"

	for i, item in enumerate(items, 1):
		question = item.get("question", "").strip()
		if not question:
			continue
		policy_name = item.get("policy_name")
		document_id = item.get("document_id")

		# Load full document context
		context_text = load_full_document(document_id, fulltext_dir)
		if max_context_chars is not None and max_context_chars > 0:
			context_text = context_text[:max_context_chars]

		prompt = build_prompt(question, policy_name, context_text)

		try:
			ans_cold, ans_hot = generate_two_answers(pipe_cold, pipe_hot, prompt)
		except Exception as e:
			print(f"Error generating for idx {i} (doc {document_id}): {e}")
			continue

		results.append(
			{
				"document_id": document_id,
				"policy_name": policy_name,
				"question": question,
				"answer_low_temp": ans_cold,
				"answer_high_temp": ans_hot,
				"temperature_low": temp_cold,
				"temperature_high": temp_hot,
				"model_id": model_id,
			}
		)

		if i % 5 == 0:
			elapsed = time.time() - start_time
			print(f"Processed {i}/{len(items)} in {elapsed:.1f}s")

	output_json.parent.mkdir(parents=True, exist_ok=True)
	with open(output_json, "w", encoding="utf-8") as f:
		json.dump(results, f, indent=2, ensure_ascii=False)

	total_elapsed = time.time() - start_time
	print(f"Saved {len(results)} pairs to {output_json} (elapsed {total_elapsed:.1f}s)")

	return {
		"count": len(results),
		"output": str(output_json),
		"elapsed_sec": total_elapsed,
	}


if __name__ == "__main__":
	import argparse
	# Defaults for your repo structure
	base_dir = Path(__file__).parent
	input_path = base_dir / "generated_questions_impl.json"
	output_path = base_dir / "dpo_pairs_impl_mistral.json"

	parser = argparse.ArgumentParser(description="Generate questions for DPO fine-tuning")
	parser.add_argument("--max_tokens", type=int, default=512,
						help="Maximum tokens for generation (default: 512)")
	parser.add_argument("--temp_cold", type=float, default=0.4,
						help="Temperature for cold generation (default: 0.4)")
	parser.add_argument("--temp_hot", type=float, default=0.7,
						help="Temperature for hot generation (default: 0.7)")
	parser.add_argument("--max_questions", type=int, default=None,
						help="Maximum number of questions to process (default: None)")
	parser.add_argument("--max_context_chars", type=int, default=None,
						help="Optionally truncate document context to this many characters (default: None = full doc)")

	args = parser.parse_args()

	# You can customize these via environment variables if desired
	model_id = "mistralai/Mistral-7B-Instruct-v0.3"
	temp_cold = args.temp_cold
	temp_hot = args.temp_hot
	max_new_tokens = args.max_tokens
	limit = args.max_questions
	max_context_chars = args.max_context_chars

	process_questions(
		input_json=input_path,
		output_json=output_path,
		model_id=model_id,
		temp_cold=temp_cold,
		temp_hot=temp_hot,
		max_new_tokens=max_new_tokens,
		limit=limit,
		max_context_chars=max_context_chars,
	)

