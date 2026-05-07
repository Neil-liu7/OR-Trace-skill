#!/usr/bin/env python3
"""Prepare OptiBench training set: run code_solutions to get gold answers,
then sample 1000 items matching OptiBench test distribution."""
import json
import random
import re
import subprocess
import tempfile
import os
from collections import defaultdict, Counter
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_code_get_answer(code: str, timeout: int = 30) -> str | None:
    """Run code_solution and extract the objective value from output."""
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
            match = re.search(r'[-+]?\d*\.?\d+(?:e[+-]?\d+)?', line.split(':')[-1] if ':' in line else line)
            if match:
                val = float(match.group())
                if val == int(val):
                    return str(int(val))
                return str(val)
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None
    finally:
        os.unlink(tf_path)


def process_item(item):
    """Process a single item: run code and return (index, answer)."""
    answer = run_code_get_answer(item['code_solution'])
    return item['question_id'], answer


def classify_for_optibench(question: str) -> str:
    """Classify problem type to match OptiBench distribution."""
    q = question.lower()
    if 'integer' in q or 'binary' in q or 'must be whole' in q or 'indivisible' in q:
        return 'integer'
    if 'nonlinear' in q or 'quadratic' in q or 'x^2' in q or 'squared' in q or 'exponential' in q:
        return 'nonlinear'
    return 'linear'


def main():
    random.seed(42)

    source_path = Path("data/benchmarks/resocratic_optibench_similar_2000.jsonl")
    test_path = Path("data/benchmarks/OptiBench.jsonl")
    output_path = Path("data/benchmarks/OptiBench_Train_1000.jsonl")

    # Load source
    with open(source_path) as f:
        items = [json.loads(l) for l in f]
    print(f"Source items: {len(items)}")

    # Get test distribution
    test_types = Counter()
    with open(test_path) as f:
        for line in f:
            d = json.loads(line)
            test_types[classify_for_optibench(d['question'])] += 1
    test_total = sum(test_types.values())
    print(f"\nOptiBench test distribution ({test_total} items):")
    for t, c in test_types.most_common():
        print(f"  {t}: {c} ({c/test_total*100:.1f}%)")

    # Run code solutions to get gold answers
    print(f"\nRunning code solutions to extract gold answers...")
    answers = {}
    failed = 0
    for i, item in enumerate(items):
        qid = item['question_id']
        ans = run_code_get_answer(item['code_solution'])
        if ans is not None:
            answers[qid] = ans
        else:
            failed += 1
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(items)}, success={len(answers)}, failed={failed}")

    print(f"\nTotal: success={len(answers)}, failed={failed}")

    # Filter to items with valid answers
    valid_items = [it for it in items if it['question_id'] in answers]
    print(f"Valid items with answers: {len(valid_items)}")

    # Classify by type (use the type field from source, map to test categories)
    by_type = defaultdict(list)
    for it in valid_items:
        src_type = it.get('type', '')
        if 'nonlinear' in src_type:
            cat = 'nonlinear'
        elif 'integer' in it['question'].lower() or 'binary' in it['question'].lower():
            cat = 'integer'
        else:
            cat = 'linear'
        by_type[cat].append(it)

    print(f"\nValid items by category:")
    for cat, items_list in by_type.items():
        print(f"  {cat}: {len(items_list)}")

    # Sample 1000 matching test distribution
    target = 1000
    sampled = []
    for cat, test_count in test_types.items():
        ratio = test_count / test_total
        n = max(10, int(target * ratio))
        available = by_type.get(cat, [])
        n = min(n, len(available))
        sampled.extend(random.sample(available, n))

    # Fill to 1000 from remaining
    sampled_ids = set(s['question_id'] for s in sampled)
    remaining = [it for it in valid_items if it['question_id'] not in sampled_ids]
    if len(sampled) < target:
        extra = random.sample(remaining, min(target - len(sampled), len(remaining)))
        sampled.extend(extra)
    if len(sampled) > target:
        sampled = random.sample(sampled, target)

    random.shuffle(sampled)
    print(f"\nSampled: {len(sampled)}")

    # Final distribution
    final_types = Counter()
    for s in sampled:
        src_type = s.get('type', '')
        if 'nonlinear' in src_type:
            final_types['nonlinear'] += 1
        else:
            final_types['linear'] += 1
    print("Final distribution:")
    for t, c in final_types.most_common():
        print(f"  {t}: {c} ({c/len(sampled)*100:.1f}%)")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for i, s in enumerate(sampled):
            record = {
                "question_id": f"OptiBench_Train_{i:04d}",
                "benchmark": "OptiBench_Train",
                "question": s["question"],
                "answer": answers[s["question_id"]],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(sampled)} items -> {output_path}")


if __name__ == "__main__":
    main()
