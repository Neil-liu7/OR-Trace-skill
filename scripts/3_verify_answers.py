#!/usr/bin/env python3
"""Step 3: Verify traces by comparing predicted answers with gold answers. Keep only correct ones."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import answers_match, extract_final_answer, load_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter traces to keep only correct ones.")
    parser.add_argument("input_file", help="Traces JSONL (from step 2)")
    parser.add_argument("output_file", help="Verified (correct) traces JSONL")
    parser.add_argument("--rejected-file", help="Optional: save rejected traces")
    args = parser.parse_args()

    rows = load_jsonl(args.input_file)
    correct = []
    rejected = []

    for row in rows:
        if row.get("status") != "success":
            rejected.append({**row, "reject_reason": "generation_error"})
            continue

        response = row.get("model_response") or row.get("raw_model_response", "")
        predicted = extract_final_answer(response)
        if not predicted:
            reasoning = row.get("model_think") or row.get("raw_reasoning", "")
            if reasoning:
                predicted = extract_final_answer(reasoning)
        gold = row.get("answer", "")

        row["predicted_answer"] = predicted
        row["is_correct"] = int(answers_match(predicted, gold))

        if row["is_correct"]:
            correct.append(row)
        else:
            row["reject_reason"] = "wrong_answer"
            rejected.append(row)

    write_jsonl(args.output_file, correct)
    print(f"Verified: {len(correct)}/{len(rows)} correct -> {args.output_file}")
    print(f"Rejected: {len(rejected)}/{len(rows)}")

    if args.rejected_file:
        write_jsonl(args.rejected_file, rejected)
        print(f"Rejected traces -> {args.rejected_file}")


if __name__ == "__main__":
    main()
