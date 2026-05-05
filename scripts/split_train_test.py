#!/usr/bin/env python3
"""Split benchmark into train/test based on LOO results with dependency constraints.

Test set = LOO-correct questions whose retrieved helpers are all in the training set.
Uses greedy algorithm: force "hub" LOO-correct questions (depended upon by others) into
training to maximize test set size.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import load_jsonl, write_jsonl


def build_split(
    benchmark_rows: list[dict],
    loo_predictions: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    """Split benchmark into train/test based on LOO dependency graph.

    Returns (train_rows, test_rows, split_info).
    """
    loo_by_qid = {r["question_id"]: r for r in loo_predictions}

    correct_ids = set(
        r["question_id"] for r in loo_predictions if r.get("is_correct") == 1
    )
    incorrect_ids = set(
        r["question_id"] for r in loo_predictions if r.get("is_correct") != 1
    )

    helpers_map: dict[str, set[str]] = {}
    for r in loo_predictions:
        if r.get("is_correct") == 1:
            helpers_map[r["question_id"]] = set(r.get("retrieved_question_ids", []))

    # Count how many LOO-correct questions depend on each LOO-correct helper
    dependency_count: Counter = Counter()
    for qid, helpers in helpers_map.items():
        for h in helpers:
            if h in correct_ids:
                dependency_count[h] += 1

    # Greedy: force most-needed LOO-correct helpers into training
    training_ids = set(incorrect_ids)
    forced_to_training = set()
    for helper, _ in dependency_count.most_common():
        forced_to_training.add(helper)
        training_ids.add(helper)

    # Check which remaining LOO-correct questions can be in test
    test_ids = set()
    for qid in correct_ids:
        if qid in forced_to_training:
            continue
        helpers = helpers_map.get(qid, set())
        if helpers <= training_ids:
            test_ids.add(qid)

    # Any remaining unresolved go to training
    remaining = correct_ids - test_ids - forced_to_training
    training_ids |= remaining

    # Build row lists preserving original order
    qid_to_row = {r["question_id"]: r for r in benchmark_rows}
    train_rows = [r for r in benchmark_rows if r["question_id"] in training_ids]
    test_rows = [r for r in benchmark_rows if r["question_id"] in test_ids]

    split_info = {
        "total": len(benchmark_rows),
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "test_ratio": len(test_rows) / max(len(benchmark_rows), 1),
        "loo_correct": len(correct_ids),
        "loo_incorrect": len(incorrect_ids),
        "forced_hub_to_training": len(forced_to_training),
        "remaining_unresolved": len(remaining),
        "test_question_ids": sorted(test_ids),
        "forced_hub_ids": sorted(forced_to_training),
    }

    return train_rows, test_rows, split_info


def verify_constraint(test_rows: list[dict], training_ids: set[str], loo_predictions: list[dict]) -> bool:
    """Verify that all test questions' helpers are in training set."""
    loo_by_qid = {r["question_id"]: r for r in loo_predictions}
    violations = []
    for row in test_rows:
        qid = row["question_id"]
        pred = loo_by_qid.get(qid, {})
        helpers = set(pred.get("retrieved_question_ids", []))
        missing = helpers - training_ids
        if missing:
            violations.append((qid, missing))
    if violations:
        print(f"ERROR: {len(violations)} constraint violations!")
        for qid, missing in violations[:5]:
            print(f"  {qid} needs {missing} not in training")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Split benchmark into train/test based on LOO results.")
    parser.add_argument("benchmark_file", help="Original benchmark JSONL")
    parser.add_argument("loo_file", help="LOO predictions JSONL (with retrieved_question_ids)")
    parser.add_argument("--output-dir", default="data/splits", help="Output directory")
    parser.add_argument("--name", help="Benchmark name (auto-detected from filename if not given)")
    args = parser.parse_args()

    benchmark_rows = load_jsonl(args.benchmark_file)
    loo_predictions = load_jsonl(args.loo_file)

    name = args.name or Path(args.benchmark_file).stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Splitting {name}: {len(benchmark_rows)} total questions")
    print(f"LOO predictions: {len(loo_predictions)} items")

    train_rows, test_rows, split_info = build_split(benchmark_rows, loo_predictions)

    # Verify constraint
    training_ids = set(r["question_id"] for r in train_rows)
    ok = verify_constraint(test_rows, training_ids, loo_predictions)

    print(f"\nSplit result:")
    print(f"  Train: {len(train_rows)} ({len(train_rows)/len(benchmark_rows)*100:.1f}%)")
    print(f"  Test:  {len(test_rows)} ({len(test_rows)/len(benchmark_rows)*100:.1f}%)")
    print(f"  Forced hubs to training: {split_info['forced_hub_to_training']}")
    print(f"  Constraint satisfied: {'YES' if ok else 'NO'}")

    write_jsonl(out_dir / f"{name}_train.jsonl", train_rows)
    write_jsonl(out_dir / f"{name}_test.jsonl", test_rows)

    with open(out_dir / f"{name}_split_info.json", "w") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print(f"\nWrote:")
    print(f"  {out_dir / f'{name}_train.jsonl'}")
    print(f"  {out_dir / f'{name}_test.jsonl'}")
    print(f"  {out_dir / f'{name}_split_info.json'}")


if __name__ == "__main__":
    main()
