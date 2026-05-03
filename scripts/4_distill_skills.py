#!/usr/bin/env python3
"""Step 4: Distill procedural skills from verified reasoning traces."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import aiohttp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from api import call_llm, now_iso
from data import extract_xml_content, load_jsonl, render_prompt, write_jsonl


def trace_from_item(item: dict) -> str:
    trace = (item.get("reasoning_trace") or "").strip()
    if trace:
        return trace
    think = (item.get("model_think") or "").strip()
    response = (item.get("model_response") or "").strip()
    return "\n\n".join(part for part in [think, response] if part)


async def process_one(
    session: aiohttp.ClientSession,
    item: dict,
    prompt_template: str,
    cfg: dict,
) -> dict:
    trace = trace_from_item(item)
    prompt = render_prompt(
        prompt_template,
        problem=item.get("question", ""),
        trace=trace,
        answer=item.get("answer", ""),
    )
    sc = cfg["skill_distillation"]

    resp = await call_llm(
        session, prompt,
        model=cfg["model"],
        temperature=sc["temperature"],
        max_tokens=sc["max_tokens"],
        timeout=sc.get("timeout", 300),
        max_retries=sc.get("max_retries", 3),
        api_base_url=cfg["api_base_url"],
        api_key=cfg["api_key"],
    )

    if resp.status != "success":
        return {
            **item,
            "distill_model": cfg["model"],
            "procedure": "",
            "worked_example": "",
            "inject_text": "",
            "keywords": "",
            "status": "error",
            "error": resp.error,
            "timestamp": now_iso(),
        }

    content = resp.text
    procedure = extract_xml_content(content, "solving_procedure")
    worked_example = extract_xml_content(content, "worked_example")
    keywords = extract_xml_content(content, "retrieval_keywords")

    inject_text = ""
    if procedure:
        inject_text = procedure
        if worked_example:
            inject_text += "\n\nWorked Example:\n" + worked_example

    return {
        **item,
        "distill_model": cfg["model"],
        "procedure": procedure,
        "worked_example": worked_example,
        "inject_text": inject_text,
        "keywords": keywords,
        "raw_skill_response": content,
        "distill_prompt_tokens": resp.prompt_tokens,
        "distill_completion_tokens": resp.completion_tokens,
        "status": "success" if procedure else "error",
        "error": "" if procedure else "missing procedure",
        "timestamp": now_iso(),
    }


async def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rows = load_jsonl(args.input_file)
    if args.limit:
        rows = rows[:args.limit]
    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")

    max_concurrent = cfg["skill_distillation"].get("max_concurrent", 8)
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
    print(f"Distilled {success}/{len(out)} skills -> {args.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill procedural skills from traces.")
    parser.add_argument("input_file", help="Verified traces JSONL")
    parser.add_argument("prompt_file", help="Skill distillation prompt")
    parser.add_argument("output_file", help="Output skills JSONL")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
