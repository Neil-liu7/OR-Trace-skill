#!/usr/bin/env python3
"""Step 6: Evaluate predictions and generate report."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import load_jsonl
from evaluate import compute_accuracy, cross_analysis, print_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction results.")
    parser.add_argument("predictions_file", help="Predictions JSONL")
    parser.add_argument("--compare", help="Optional: second predictions file for cross-analysis")
    parser.add_argument("--label", default="", help="Label for the report")
    parser.add_argument("--compare-label", default="Comparison", help="Label for comparison")
    args = parser.parse_args()

    results = load_jsonl(args.predictions_file)
    print_report(results, args.label or args.predictions_file)

    if args.compare:
        compare_results = load_jsonl(args.compare)
        print()
        print_report(compare_results, args.compare_label)
        print()

        cross = cross_analysis(results, compare_results)
        print(f"=== Cross Analysis: {args.label} vs {args.compare_label} ===")
        print(f"Both correct:  {cross['both_correct']}")
        print(f"A only:        {cross['a_only']}")
        print(f"B only:        {cross['b_only']}")
        print(f"Both wrong:    {cross['both_wrong']}")


if __name__ == "__main__":
    main()
