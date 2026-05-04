#!/usr/bin/env python3
"""Step 7: Lamarckian Skill Evolution — iteratively improve skills via inference feedback."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import aiohttp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from api import call_llm, now_iso
from data import (
    answers_match,
    extract_final_answer,
    extract_xml_content,
    load_jsonl,
    render_prompt,
    write_jsonl,
)
from retrieval import SkillBank


FALLBACK_PROMPT = """You are a helpful assistant that solves problems. Show each step with its computation, then give the final answer.

Problem:
{PROBLEM}"""


# ---------------------------------------------------------------------------
# Phase 1: Inference with current skill bank
# ---------------------------------------------------------------------------

async def _infer_one(
    session: aiohttp.ClientSession,
    item: dict,
    bank: SkillBank,
    prompt_template: str,
    cfg: dict,
) -> dict:
    ic = cfg["inference"]
    question = item.get("question", "")
    top_k = ic.get("top_k", 2)

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
        hits = bank.hybrid_search(question, top_k, rrf_k=rc.get("rrf_k", 60))
    elif search_method == "structural_hybrid":
        hits = bank.structural_hybrid_search(question, top_k, rrf_k=rc.get("rrf_k", 60))
    else:
        hits = bank.search(question, top_k)

    if hits:
        hints = "\n\n---\n\n".join(h["skill_text"] for h in hits)
        prompt = render_prompt(prompt_template, problem=question, hints=hints)
    else:
        prompt = FALLBACK_PROMPT.replace("{PROBLEM}", question)

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
        return {**item, "status": "error", "error": resp.error}

    predicted = extract_final_answer(resp.text)
    gold = item.get("answer", "")

    return {
        **item,
        "raw_reasoning": resp.reasoning,
        "raw_model_response": resp.text,
        "predicted_answer": predicted,
        "is_correct": int(answers_match(predicted, gold)) if gold else 0,
        "status": "success",
    }


async def infer_with_skills(
    session: aiohttp.ClientSession,
    items: list[dict],
    bank: SkillBank,
    prompt_template: str,
    cfg: dict,
) -> list[dict]:
    max_concurrent = cfg["inference"].get("max_concurrent", 8)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def guarded(item: dict) -> dict:
        async with semaphore:
            return await _infer_one(session, item, bank, prompt_template, cfg)

    return list(await asyncio.gather(*(guarded(item) for item in items)))


# ---------------------------------------------------------------------------
# Phase 2: Re-distill skills from correct traces (Lamarckian)
# ---------------------------------------------------------------------------

def _build_trace(item: dict) -> str:
    reasoning = (item.get("raw_reasoning") or "").strip()
    response = (item.get("raw_model_response") or "").strip()
    if reasoning and response:
        return f"{reasoning}\n\n{response}"
    return reasoning or response


async def _redistill_one(
    session: aiohttp.ClientSession,
    item: dict,
    prompt_template: str,
    cfg: dict,
    evolution_round: int,
) -> dict:
    trace = _build_trace(item)
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
        return {**item, "status": "error", "error": resp.error}

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
        "procedure": procedure,
        "worked_example": worked_example,
        "inject_text": inject_text,
        "keywords": keywords,
        "raw_skill_response": content,
        "evolution_round": evolution_round,
        "status": "success" if procedure else "error",
        "error": "" if procedure else "missing procedure",
        "timestamp": now_iso(),
    }


async def redistill_skills(
    session: aiohttp.ClientSession,
    correct_items: list[dict],
    prompt_template: str,
    cfg: dict,
    evolution_round: int,
) -> list[dict]:
    max_concurrent = cfg["skill_distillation"].get("max_concurrent", 8)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def guarded(item: dict) -> dict:
        async with semaphore:
            return await _redistill_one(session, item, prompt_template, cfg, evolution_round)

    return list(await asyncio.gather(*(guarded(item) for item in correct_items)))


# ---------------------------------------------------------------------------
# Phase 3: Merge skill banks
# ---------------------------------------------------------------------------

def merge_skills(old_skills: list[dict], new_skills: list[dict]) -> list[dict]:
    """Merge old and new skills. For the same question_id, keep the newer version."""
    by_qid: dict[str, dict] = {}
    for s in old_skills:
        qid = s.get("question_id", "")
        if qid:
            by_qid[qid] = s
    for s in new_skills:
        if s.get("status") != "success":
            continue
        qid = s.get("question_id", "")
        if qid:
            by_qid[qid] = s
    return list(by_qid.values())


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evo_cfg = cfg.get("evolution", {})
    rounds = args.rounds or evo_cfg.get("rounds", 3)

    items = load_jsonl(args.data_file)
    if args.limit:
        items = items[: args.limit]

    current_skills = load_jsonl(args.skill_file)
    infer_prompt = Path(args.infer_prompt).read_text(encoding="utf-8")
    redistill_prompt = Path(args.redistill_prompt).read_text(encoding="utf-8")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = cfg["inference"].get("retrieval", {})
    embed_model = (
        rc.get("embed_model")
        if rc.get("method") in ("hybrid", "structural_hybrid")
        else None
    )

    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=cfg["inference"].get("max_concurrent", 8) * 2)

    prev_accuracy = None

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for rnd in range(rounds):
            round_dir = out_dir / f"round_{rnd}"
            round_dir.mkdir(parents=True, exist_ok=True)

            valid_skills = [s for s in current_skills if (s.get("inject_text") or s.get("procedure", "")).strip()]
            bank = SkillBank(valid_skills, embed_model=embed_model)
            print(f"\n{'='*60}")
            print(f"  Evolution Round {rnd} — {len(bank.records)} skills in bank")
            print(f"{'='*60}")

            # --- Infer ---
            print(f"  [1/3] Inferring on {len(items)} items...")
            predictions = await infer_with_skills(session, items, bank, infer_prompt, cfg)
            write_jsonl(round_dir / "predictions.jsonl", predictions)

            correct = [p for p in predictions if p.get("is_correct") == 1]
            total = sum(1 for p in predictions if p.get("status") == "success")
            accuracy = len(correct) / max(total, 1)
            print(f"  Accuracy: {len(correct)}/{total} = {accuracy * 100:.1f}%")

            # --- Early stop check ---
            min_imp = evo_cfg.get("min_improvement", 0.0)
            if prev_accuracy is not None and accuracy < prev_accuracy - min_imp:
                print(f"  Accuracy dropped ({prev_accuracy*100:.1f}% -> {accuracy*100:.1f}%), stopping evolution.")
                break
            prev_accuracy = accuracy

            if not correct:
                print("  No correct predictions, skipping re-distillation.")
                continue

            # --- Re-distill ---
            print(f"  [2/3] Re-distilling {len(correct)} correct traces...")
            new_skills = await redistill_skills(session, correct, redistill_prompt, cfg, evolution_round=rnd + 1)
            write_jsonl(round_dir / "new_skills.jsonl", new_skills)

            success_skills = [s for s in new_skills if s.get("status") == "success"]
            print(f"  Distilled {len(success_skills)}/{len(correct)} new skills")

            # --- Merge ---
            print(f"  [3/3] Merging skills...")
            current_skills = merge_skills(current_skills, new_skills)
            write_jsonl(round_dir / "merged_skills.jsonl", current_skills)
            print(f"  Merged bank size: {len(current_skills)}")

    # --- Write final output ---
    final_path = out_dir / "evolved_skills.jsonl"
    write_jsonl(final_path, current_skills)
    print(f"\nEvolution complete. Final skill bank: {len(current_skills)} skills -> {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lamarckian Skill Evolution: iteratively improve skills via inference feedback.",
    )
    parser.add_argument("data_file", help="Training problems JSONL")
    parser.add_argument("skill_file", help="Initial skill library JSONL")
    parser.add_argument("--infer-prompt", default="prompts/skill_infer.txt", help="Inference prompt template")
    parser.add_argument("--redistill-prompt", default="prompts/skill_redistill.txt", help="Re-distillation prompt")
    parser.add_argument("--output-dir", default="outputs/evolution", help="Output directory for evolution rounds")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rounds", type=int, help="Number of evolution rounds (overrides config)")
    parser.add_argument("--limit", type=int, help="Limit number of problems")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
