#!/usr/bin/env python3
"""Step 5: BM25 retrieval + No-Think inference with procedural skills."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import aiohttp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from api import call_llm, now_iso
from data import answers_match, extract_final_answer, load_jsonl, render_prompt, write_jsonl
from retrieval import SkillBank


FALLBACK_PROMPT = """You are a helpful assistant that solves problems. Show each step with its computation, then give the final answer.

Problem:
{PROBLEM}"""


async def process_one(
    session: aiohttp.ClientSession,
    item: dict,
    bank: SkillBank,
    prompt_template: str,
    cfg: dict,
) -> dict:
    ic = cfg["inference"]
    question = item.get("question", "")
    top_k = ic.get("top_k", 3)

    rc = ic.get("retrieval", {})
    search_method = rc.get("method", "bm25")

    if search_method == "structural":
        hits = bank.structural_search(question, top_k)
    elif search_method == "adaptive":
        hits = bank.adaptive_search(
            question, top_k,
            min_score=rc.get("min_score", 40.0),
            top1_ratio=rc.get("top1_ratio", 1.5),
            confidence_threshold=rc.get("confidence_threshold", 50.0),
            ratio_cutoff=rc.get("ratio_cutoff", 0.7),
        )
    elif search_method == "hybrid":
        hits = bank.hybrid_search(
            question, top_k,
            rrf_k=rc.get("rrf_k", 60),
        )
    elif search_method == "structural_hybrid":
        hits = bank.structural_hybrid_search(
            question, top_k,
            rrf_k=rc.get("rrf_k", 60),
        )
    else:
        hits = bank.search(question, top_k)

    if hits:
        hints = "\n\n---\n\n".join(h["skill_text"] for h in hits)
        prompt = render_prompt(prompt_template, problem=question, hints=hints)
        if len(hits) == 1:
            retrieval_mode = "skill_top1"
        else:
            retrieval_mode = "skill_topk"
    else:
        prompt = FALLBACK_PROMPT.replace("{PROBLEM}", question)
        retrieval_mode = "fallback"

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
            "mode": "skill_retrieval",
            "retrieval_mode": retrieval_mode,
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
        "mode": "skill_retrieval",
        "retrieval_mode": retrieval_mode,
        "raw_reasoning": resp.reasoning,
        "raw_model_response": resp.text,
        "predicted_answer": predicted,
        "is_correct": int(answers_match(predicted, gold)) if gold else None,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "total_tokens": resp.total_tokens,
        "retrieved_count": len(hits),
        "retrieved_question_ids": [h["question_id"] for h in hits],
        "retrieved_scores": [round(h["score"], 4) for h in hits],
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
    skills = load_jsonl(args.skill_file)
    rc = cfg["inference"].get("retrieval", {})
    embed_model = rc.get("embed_model") if rc.get("method") in ("hybrid", "structural_hybrid") else None
    bank = SkillBank(skills, embed_model=embed_model)
    print(f"Loaded {len(bank.records)} skills into retrieval bank")

    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")
    max_concurrent = cfg["inference"].get("max_concurrent", 8)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def guarded(session, item):
        async with semaphore:
            return await process_one(session, item, bank, prompt_template, cfg)

    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=max_concurrent * 2)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        out = await asyncio.gather(*(guarded(session, item) for item in rows))

    write_jsonl(args.output_file, out)
    total = len(out)
    correct = sum(1 for r in out if r.get("is_correct") == 1)
    print(f"Wrote {total} predictions -> {args.output_file}")
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    from collections import Counter
    modes = Counter(r.get("retrieval_mode", "unknown") for r in out)
    print(f"Retrieval modes: {dict(modes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 retrieval + No-Think inference.")
    parser.add_argument("input_file", help="Test problems JSONL")
    parser.add_argument("skill_file", help="Skill library JSONL")
    parser.add_argument("prompt_file", help="Inference prompt")
    parser.add_argument("output_file", help="Output predictions JSONL")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
