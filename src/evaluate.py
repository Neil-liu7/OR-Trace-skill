"""Evaluation utilities: accuracy computation, cross-analysis, reporting."""
from __future__ import annotations

from typing import Any


def compute_accuracy(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct") == 1)
    errors = sum(1 for r in results if r.get("status") == "error")
    avg_tokens = sum(r.get("completion_tokens", 0) for r in results) / max(total, 1)
    return {
        "total": total,
        "correct": correct,
        "errors": errors,
        "accuracy": correct / max(total, 1),
        "avg_completion_tokens": avg_tokens,
    }


def cross_analysis(
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]],
    key: str = "question",
) -> dict[str, int]:
    b_map = {r.get(key, "")[:150]: r.get("is_correct") == 1 for r in results_b}
    both_correct = a_only = b_only = both_wrong = 0
    for r in results_a:
        k = r.get(key, "")[:150]
        a_ok = r.get("is_correct") == 1
        b_ok = b_map.get(k, False)
        if a_ok and b_ok:
            both_correct += 1
        elif a_ok:
            a_only += 1
        elif b_ok:
            b_only += 1
        else:
            both_wrong += 1
    return {
        "both_correct": both_correct,
        "a_only": a_only,
        "b_only": b_only,
        "both_wrong": both_wrong,
    }


def print_report(results: list[dict[str, Any]], label: str = "") -> None:
    stats = compute_accuracy(results)
    header = f"=== {label} ===" if label else "=== Results ==="
    print(header)
    print(f"Total: {stats['total']}")
    print(f"Correct: {stats['correct']}/{stats['total']} = {stats['accuracy']*100:.1f}%")
    print(f"Errors: {stats['errors']}")
    print(f"Avg completion tokens: {stats['avg_completion_tokens']:.0f}")
