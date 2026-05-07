#!/usr/bin/env python3
"""Prepare OptMATH training set: sample 500 items from OptMATH_Train, excluding test set overlap."""
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def classify_problem(text: str) -> str:
    t = text.lower()
    if "schedul" in t or "makespan" in t or "job" in t and "machine" in t:
        return "scheduling"
    if "transport" in t or "shipping" in t or "vehicle" in t or "route" in t:
        return "transport"
    if "assign" in t or "allocation" in t:
        return "assignment"
    if "network" in t or "flow" in t or "graph" in t:
        return "network_flow"
    if "facility" in t or "location" in t or "warehouse" in t:
        return "facility"
    if "knapsack" in t or "packing" in t or "bin" in t:
        return "knapsack"
    return "LP"


def main():
    random.seed(42)

    test_path = Path("data/benchmarks/OptMATH_Bench_166.jsonl")
    train_path = Path("data/benchmarks/OptMATH_Train.jsonl")
    output_path = Path("data/benchmarks/OptMATH_Train_500.jsonl")

    # Load test set fingerprints for dedup
    test_fps = set()
    with open(test_path) as f:
        for line in f:
            d = json.loads(line)
            test_fps.add(d["question"].strip()[:500])

    # Load and filter training data
    candidates = []
    with open(train_path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            inp = d.get("input", "").strip()
            if inp.startswith("# Question:"):
                inp = inp[len("# Question:"):].strip()

            # Skip if overlaps with test set
            if inp[:500] in test_fps:
                continue

            # Skip very long or very short problems
            if len(inp) < 200 or len(inp) > 8000:
                continue

            candidates.append({
                "index": i,
                "question": inp,
                "answer": d.get("answer", ""),
                "category": classify_problem(inp),
            })

    print(f"Candidates after filtering: {len(candidates)}")

    # Stratified sampling by category
    by_category = defaultdict(list)
    for c in candidates:
        by_category[c["category"]].append(c)

    print("\nCategory distribution:")
    for cat, items in sorted(by_category.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(items)}")

    # Sample proportionally, total 500
    target = 500
    total = len(candidates)
    sampled = []

    for cat, items in by_category.items():
        n = max(10, int(target * len(items) / total))
        n = min(n, len(items))
        sampled.extend(random.sample(items, n))

    # If under 500, fill from remaining
    sampled_indices = set(s["index"] for s in sampled)
    remaining = [c for c in candidates if c["index"] not in sampled_indices]
    if len(sampled) < target:
        extra = random.sample(remaining, min(target - len(sampled), len(remaining)))
        sampled.extend(extra)

    # If over 500, trim
    if len(sampled) > target:
        sampled = random.sample(sampled, target)

    print(f"\nSampled: {len(sampled)}")

    # Final category distribution
    final_cats = defaultdict(int)
    for s in sampled:
        final_cats[s["category"]] += 1
    print("Final distribution:")
    for cat, n in sorted(final_cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}")

    # Write output in pipeline format
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for i, s in enumerate(sampled):
            record = {
                "question_id": f"OptMATH_Train_{i:04d}",
                "benchmark": "OptMATH_Train",
                "question": s["question"],
                "answer": s["answer"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(sampled)} items -> {output_path}")


if __name__ == "__main__":
    main()
