#!/usr/bin/env python3
"""Step 1: Synthesize new problems from seed examples."""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

import aiohttp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from api import LLMResponse, call_llm, now_iso
from data import extract_xml_content, load_jsonl, render_prompt, write_jsonl


async def synthesize_one(
    session: aiohttp.ClientSession,
    seed_examples: list[dict],
    idx: int,
    prompt_template: str,
    cfg: dict,
) -> dict | None:
    examples_block = "\n\n".join(
        f"Problem {i+1}:\n{ex['question']}\nAnswer: {ex.get('answer', '?')}"
        for i, ex in enumerate(seed_examples)
    )
    prompt = render_prompt(prompt_template, examples=examples_block)

    resp = await call_llm(
        session, prompt,
        model=cfg["model"],
        temperature=cfg["synthesis"]["temperature"],
        max_tokens=cfg["synthesis"]["max_tokens"],
        api_base_url=cfg["api_base_url"],
        api_key=cfg["api_key"],
    )
    if resp.status != "success":
        return None

    problem = extract_xml_content(resp.text, "problem")
    answer = extract_xml_content(resp.text, "answer")
    if not problem or not answer:
        return None

    return {
        "question_id": f"synth_{idx:04d}",
        "benchmark": "synthetic",
        "question": problem,
        "answer": answer,
        "source": "synthesized",
        "timestamp": now_iso(),
    }


async def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed_data = load_jsonl(args.seed_file)
    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")

    num_per_seed = cfg["synthesis"].get("num_problems_per_seed", 5)
    max_concurrent = cfg["synthesis"].get("max_concurrent", 4)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def guarded(session, seeds, idx):
        async with semaphore:
            return await synthesize_one(session, seeds, idx, prompt_template, cfg)

    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=max_concurrent * 2)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        idx = 0
        for _ in range(num_per_seed):
            for seed_batch_start in range(0, len(seed_data), 3):
                seeds = seed_data[seed_batch_start:seed_batch_start + 3]
                if seeds:
                    tasks.append(guarded(session, seeds, idx))
                    idx += 1

        if args.limit:
            tasks = tasks[:args.limit]

        results = await asyncio.gather(*tasks)

    out = [r for r in results if r is not None]
    write_jsonl(args.output_file, out)
    print(f"Synthesized {len(out)}/{len(tasks)} problems -> {args.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize new problems from seed examples.")
    parser.add_argument("seed_file", help="Seed problems JSONL")
    parser.add_argument("prompt_file", help="Synthesis prompt template")
    parser.add_argument("output_file", help="Output synthesized problems JSONL")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, help="Max number of synthesis calls")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
