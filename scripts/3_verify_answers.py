#!/usr/bin/env python3
"""Step 3: Verify traces by comparing predicted answers with gold answers. Keep only correct ones.

Supports two modes:
1. Direct extraction: parse numerical answer from model text
2. Code execution: if model outputs code (e.g. gurobipy), execute it and extract answer from stdout
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import answers_match, extract_final_answer, load_jsonl, write_jsonl


def extract_code_block(text: str) -> str | None:
    """Extract Python code block from markdown-formatted response."""
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
    return None


def run_code_get_answer(code: str, timeout: int = 30) -> str | None:
    """Execute code and extract the objective/answer from stdout."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
        tf.write(code)
        tf_path = tf.name
    try:
        result = subprocess.run(
            ['python3', tf_path], capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return None
        output = result.stdout.strip()
        if not output:
            return None
        lines = output.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if not line or line.startswith('-'):
                continue
            val_part = line.split(':')[-1] if ':' in line else line
            match = re.search(r'[-+]?\d*\.?\d+(?:e[+-]?\d+)?', val_part)
            if match:
                val_str = match.group()
                try:
                    val = float(val_str)
                    if val == int(val) and 'e' not in val_str.lower():
                        return str(int(val))
                    return val_str
                except ValueError:
                    continue
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None
    finally:
        os.unlink(tf_path)


def get_predicted_answer(row: dict) -> str:
    """Try multiple strategies to extract answer from a trace."""
    response = row.get("model_response") or row.get("raw_model_response", "")
    reasoning = row.get("model_think") or row.get("raw_reasoning", "")

    # Strategy 1: direct extraction from response text
    predicted = extract_final_answer(response)
    if predicted:
        return predicted

    # Strategy 2: direct extraction from reasoning
    if reasoning:
        predicted = extract_final_answer(reasoning)
        if predicted:
            return predicted

    # Strategy 3: execute code from response
    code = extract_code_block(response)
    if code:
        predicted = run_code_get_answer(code)
        if predicted:
            return predicted

    # Strategy 4: execute code from reasoning
    if reasoning:
        code = extract_code_block(reasoning)
        if code:
            predicted = run_code_get_answer(code)
            if predicted:
                return predicted

    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter traces to keep only correct ones.")
    parser.add_argument("input_file", help="Traces JSONL (from step 2)")
    parser.add_argument("output_file", help="Verified (correct) traces JSONL")
    parser.add_argument("--rejected-file", help="Optional: save rejected traces")
    args = parser.parse_args()

    rows = load_jsonl(args.input_file)
    correct = []
    rejected = []

    for i, row in enumerate(rows):
        if row.get("status") != "success":
            rejected.append({**row, "reject_reason": "generation_error"})
            continue

        predicted = get_predicted_answer(row)
        gold = str(row.get("answer", ""))

        row["predicted_answer"] = predicted
        row["is_correct"] = int(answers_match(predicted, gold))

        if row["is_correct"]:
            correct.append(row)
        else:
            row["reject_reason"] = "wrong_answer"
            rejected.append(row)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(rows)}, correct so far: {len(correct)}")

    write_jsonl(args.output_file, correct)
    print(f"Verified: {len(correct)}/{len(rows)} correct -> {args.output_file}")
    print(f"Rejected: {len(rejected)}/{len(rows)}")

    if args.rejected_file:
        write_jsonl(args.rejected_file, rejected)
        print(f"Rejected traces -> {args.rejected_file}")


if __name__ == "__main__":
    main()
