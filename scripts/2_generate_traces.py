#!/usr/bin/env python3
"""Step 2: Generate Think-mode reasoning traces for problems."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import aiohttp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from api import call_llm, now_iso
from data import load_jsonl, render_prompt, write_jsonl


async def process_one(
    session: aiohttp.ClientSession,
    item: dict,
    prompt_template: str,
    cfg: dict,
) -> dict:
    prompt = render_prompt(prompt_template, problem=item.get("question", ""))
    tc = cfg["trace_generation"]

    resp = await call_llm(
        session, prompt,
        model=cfg["model"],
        temperature=tc["temperature"],
        max_tokens=tc["max_tokens"],
        timeout=tc.get("timeout", 600),
        max_retries=tc.get("max_retries", 3),
        api_base_url=cfg["api_base_url"],
        api_key=cfg["api_key"],
    )

    if resp.status != "success":
        return {
            **item,
            "source_model": cfg["model"],
            "model_think": "",
            "model_response": "",
            "status": "error",
            "error": resp.error,
            "timestamp": now_iso(),
        }

    reasoning_trace = ""
    if resp.reasoning and resp.text:
        reasoning_trace = f"<think>\n{resp.reasoning}\n</think>\n\n{resp.text}"
    else:
        reasoning_trace = resp.reasoning or resp.text

    return {
        **item,
        "source_model": cfg["model"],
        "model_think": resp.reasoning,
        "model_response": resp.text,
        "reasoning_trace": reasoning_trace,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "total_tokens": resp.total_tokens,
        "status": "success" if reasoning_trace else "error",
        "error": "" if reasoning_trace else "empty trace",
        "timestamp": now_iso(),
    }


async def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rows = load_jsonl(args.input_file)
    if args.limit:
        rows = rows[:args.limit]
    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")

    max_concurrent = cfg["trace_generation"].get("max_concurrent", 8)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def guarded(session, item):
        async with semaphore:
            return await process_one(session, item, prompt_template, cfg)

    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=max_concurrent * 2)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        out = await asyncio.gather(*(guarded(session, item) for item in rows))

    write_jsonl(args.output_file, out)
    success = sum(1 for r in out if r["status"] == "success")
    print(f"Generated {success}/{len(out)} traces -> {args.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Think-mode reasoning traces.")
    parser.add_argument("input_file", help="Problems JSONL")
    parser.add_argument("prompt_file", help="Source CoT prompt")
    parser.add_argument("output_file", help="Output traces JSONL")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
