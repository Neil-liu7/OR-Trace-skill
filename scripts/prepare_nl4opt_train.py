#!/usr/bin/env python3
"""Prepare NL4OPT training set: convert from annotated format to pipeline format."""
import json
from pathlib import Path


def main():
    input_path = Path("data/benchmarks/NL4OPT_train_with_gold.jsonl")
    output_path = Path("data/benchmarks/NL4OPT_Train.jsonl")

    records = []
    with open(input_path) as f:
        for line in f:
            d = json.loads(line)
            for item_id, item in d.items():
                if item.get("verification_status") not in ("correct", "corrected"):
                    continue
                records.append({
                    "question_id": f"NL4OPT_Train_{len(records):04d}",
                    "benchmark": "NL4OPT_Train",
                    "question": item["document"],
                    "answer": str(item["gold_answer"]),
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} items -> {output_path}")


if __name__ == "__main__":
    main()
