import argparse
import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
import instructor
import httpx
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.metrics.collections import AnswerAccuracy

DEFAULT_INPUT = Path(__file__).resolve().parent / "system_eval_chunks_finetuned_answers_base.json"
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
FAITHFULNESS_MAX_TOKENS = int(os.getenv("FAITHFULNESS_MAX_TOKENS", "4096"))

base_client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=VLLM_BASE_URL,
)
model = "mistralai/Mistral-7B-Instruct-v0.3"
client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)


def build_scorer(max_output_tokens: int) -> AnswerAccuracy:
    llm = InstructorLLM(
        client=client,
        model=model,
        provider="openai",
        model_args=InstructorModelArgs(max_tokens=max_output_tokens),
    )
    return AnswerAccuracy(llm=llm)


async def wait_for_vllm(timeout_seconds: int = 300, poll_interval: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            try:
                response = await client.get(f"{VLLM_BASE_URL}/models")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass

            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(f"Timed out waiting for vLLM at {VLLM_BASE_URL}")

            await asyncio.sleep(poll_interval)

def load_items(input_path: Path) -> list[dict]:
    if input_path.suffix == ".jsonl":
        items = []
        with input_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {input_path}")
    return data
async def evaluate_file(
    input_path: Path,
    scorer: AnswerAccuracy,
    limit: int | None = None,
) -> None:
    await wait_for_vllm()

    items = load_items(input_path)

    if limit is not None:
        items = items[:limit]

    total_score = 0.0
    valid_scores = 0
    attempted = 0

    for index, item in enumerate(items, start=1):
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        reference = (item.get("reference_answer") or item.get("reference") or "").strip()
        if not question or not answer or not reference:
            print(f"[{index}] skipped: missing question, answer, or reference")
            continue

        try:
            result = await scorer.ascore(
                user_input=question,
                response=answer,
                reference=reference,
            )
        except Exception as error:
            print(f"[{index}] skipped: {error}")
            continue

        attempted += 1

        if result.value is None or result.value != result.value:
            print(f"[{index}] score=nan")
        else:
            total_score += result.value
            valid_scores += 1
            print(f"[{index}] score={result.value:.4f}")

    if valid_scores == 0:
        print("Average Answer Accuracy: nan (no valid scores)")
        return

    average_score = total_score / valid_scores
    print(f"Average Answer Accuracy: {average_score:.6f}")
    print(f"Valid scores: {valid_scores}/{attempted}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute average Answer Accuracy for a JSON or JSONL file")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to input JSON or JSONL file")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=FAITHFULNESS_MAX_TOKENS,
        help="Maximum completion tokens for the faithfulness judge model",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of items to evaluate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scorer = build_scorer(args.max_output_tokens)
    print(f"Using model: {model}")
    asyncio.run(evaluate_file(args.input, scorer=scorer, limit=args.limit))


