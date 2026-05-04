#!/usr/bin/env python3
"""Baseline: No-Think inference without skill retrieval."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import aiohttp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from api import call_llm, now_iso
from data import answers_match, extract_final_answer, load_jsonl, write_jsonl


BASELINE_PROMPT = """You are a helpful assistant that solves problems. Show each step with its computation, then give the final answer.

Problem:
{PROBLEM}"""


async def process_one(
    session: aiohttp.ClientSession,
    item: dict,
    cfg: dict,
) -> dict:
    ic = cfg["inference"]
    question = item.get("question", "")
    prompt = BASELINE_PROMPT.replace("{PROBLEM}", question)

    enable_thinking = False if ic.get("no_think", True) else None

    resp = await call_llm(
        session, prompt,
        model=cfg["model"],
        temperature=ic["temperature"],
        max_tokens=ic["max_tokens"],
        enable_thinking=enable_thinking,
        timeout=ic.get("timeout", 600),
        max_retries=ic.get("max_retries", 3),
        api_base_url=cfg["api_base_url"],
        api_key=cfg["api_key"],
    )

    if resp.status != "success":
        return {
            **item,
            "gen_model": cfg["model"],
            "mode": "baseline_no_think",
            "status": "error",
            "error": resp.error,
            "timestamp": now_iso(),
        }

    predicted = extract_final_answer(resp.text)
    if not predicted and resp.reasoning:
        predicted = extract_final_answer(resp.reasoning)
    gold = item.get("answer", "")

    return {
        **item,
        "gen_model": cfg["model"],
        "mode": "baseline_no_think",
        "raw_reasoning": resp.reasoning,
        "raw_model_response": resp.text,
        "predicted_answer": predicted,
        "is_correct": int(answers_match(predicted, gold)) if gold else None,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "total_tokens": resp.total_tokens,
        "status": "success",
        "error": "",
        "timestamp": now_iso(),
    }


async def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rows = load_jsonl(args.input_file)
    if args.limit:
        rows = rows[:args.limit]

    max_concurrent = cfg["inference"].get("max_concurrent", 8)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def guarded(session, item):
        async with semaphore:
            return await process_one(session, item, cfg)

    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=max_concurrent * 2)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        out = await asyncio.gather(*(guarded(session, item) for item in rows))

    write_jsonl(args.output_file, out)
    total = len(out)
    correct = sum(1 for r in out if r.get("is_correct") == 1)
    print(f"Wrote {total} predictions -> {args.output_file}")
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline: No-Think without skills.")
    parser.add_argument("input_file", help="Test problems JSONL")
    parser.add_argument("output_file", help="Output predictions JSONL")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
