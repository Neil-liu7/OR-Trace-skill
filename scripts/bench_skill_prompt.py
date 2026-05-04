#!/usr/bin/env python3
"""Benchmark: No-Think inference with a system-level skill prompt vs plain baseline.

Usage:
    python3 scripts/bench_skill_prompt.py <data_file> <output_file> --skill-prompt skills/skill_combined.md
    python3 scripts/bench_skill_prompt.py <data_file> <output_file>   # baseline (no skill)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from api import LLMResponse, _parse_response, now_iso
from data import answers_match, extract_final_answer, load_jsonl, write_jsonl


BASELINE_USER = """Solve the following problem. Show each step with its computation, then give the final answer.

Problem:
{PROBLEM}"""


async def call_llm_with_system(
    session: aiohttp.ClientSession,
    system: str | None,
    user: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool | None = None,
    timeout: int = 600,
    api_base_url: str = "",
    api_key: str = "",
    max_retries: int = 3,
) -> LLMResponse:
    base_url = api_base_url or os.environ.get("API_BASE_URL", "http://localhost:8001/v1/chat/completions")
    key = api_key or os.environ.get("API_KEY", "dummy")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json; charset=utf-8"}
    req_timeout = aiohttp.ClientTimeout(total=timeout)

    for attempt in range(max_retries + 1):
        try:
            async with session.post(base_url, headers=headers, json=payload, timeout=req_timeout) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    if attempt < max_retries:
                        await asyncio.sleep(min(20.0, 2.0 ** attempt))
                        continue
                    return LLMResponse(status="error", error=f"HTTP {resp.status}: {body[:300]}")
                return _parse_response(json.loads(body), user)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
            if attempt < max_retries:
                await asyncio.sleep(min(20.0, 2.0 ** attempt))
                continue
            return LLMResponse(status="error", error=str(exc))

    return LLMResponse(status="error", error="max retries exceeded")


async def process_one(
    session: aiohttp.ClientSession,
    item: dict,
    system_prompt: str | None,
    cfg: dict,
) -> dict:
    ic = cfg["inference"]
    question = item.get("question", "")
    user_msg = BASELINE_USER.replace("{PROBLEM}", question)
    enable_thinking = False if ic.get("no_think", True) else None

    resp = await call_llm_with_system(
        session, system_prompt, user_msg,
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
        return {**item, "gen_model": cfg["model"], "mode": "skill_prompt", "status": "error", "error": resp.error, "timestamp": now_iso()}

    full_text = resp.text or resp.reasoning
    predicted = extract_final_answer(full_text)
    gold = item.get("answer", "")

    return {
        **item,
        "gen_model": cfg["model"],
        "mode": "skill_prompt" if system_prompt else "baseline_no_think",
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
        rows = rows[: args.limit]

    system_prompt = None
    if args.skill_prompt:
        system_prompt = Path(args.skill_prompt).read_text(encoding="utf-8")

    max_concurrent = cfg["inference"].get("max_concurrent", 8)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def guarded(session, item):
        async with semaphore:
            return await process_one(session, item, system_prompt, cfg)

    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=max_concurrent * 2)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        out = await asyncio.gather(*(guarded(session, item) for item in rows))

    write_jsonl(args.output_file, out)
    total = len(out)
    correct = sum(1 for r in out if r.get("is_correct") == 1)
    errors = sum(1 for r in out if r.get("status") == "error")
    print(f"  {total} items, {correct} correct, {errors} errors => {correct/max(total,1)*100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="No-Think inference with optional system skill prompt.")
    parser.add_argument("input_file", help="Problems JSONL")
    parser.add_argument("output_file", help="Output predictions JSONL")
    parser.add_argument("--skill-prompt", help="Path to skill system prompt (e.g. skills/skill_combined.md)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
