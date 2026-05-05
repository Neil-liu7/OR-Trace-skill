#!/usr/bin/env python3
"""Filter and merge skills based on LOO evaluation metrics.

Computes per-skill NetScore from LOO predictions vs baseline, then:
1. Filters out harmful skills (NetScore < 0)
2. Filters out high-frequency useless skills (NetHelpRate == 0, retrieval_count >= threshold)
3. Merges similar remaining skills via embedding clustering (cosine > 0.85)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import load_jsonl, write_jsonl


def compute_skill_metrics(
    skills: list[dict],
    loo_predictions: list[dict],
    baseline_predictions: list[dict],
) -> dict[str, dict]:
    """Compute per-skill effectiveness metrics.

    Returns {question_id: {retrieval_count, help_count, hurt_count, net_help_rate, hurt_rate, net_score}}.
    """
    baseline_correct = {}
    for r in baseline_predictions:
        baseline_correct[r["question_id"]] = r.get("is_correct") == 1

    loo_correct = {}
    loo_retrieved = {}
    for r in loo_predictions:
        qid = r["question_id"]
        loo_correct[qid] = r.get("is_correct") == 1
        loo_retrieved[qid] = r.get("retrieved_question_ids", [])

    # Count per skill: how many times retrieved, how many times helped, how many times hurt
    skill_ids = set(s["question_id"] for s in skills if s.get("status") == "success")
    metrics: dict[str, dict] = {sid: {"retrieval_count": 0, "help_count": 0, "hurt_count": 0} for sid in skill_ids}

    for qid, retrieved_ids in loo_retrieved.items():
        base_ok = baseline_correct.get(qid, False)
        loo_ok = loo_correct.get(qid, False)

        for skill_qid in retrieved_ids:
            if skill_qid not in metrics:
                continue
            metrics[skill_qid]["retrieval_count"] += 1

            if not base_ok and loo_ok:
                metrics[skill_qid]["help_count"] += 1
            elif base_ok and not loo_ok:
                metrics[skill_qid]["hurt_count"] += 1

    for sid, m in metrics.items():
        rc = max(m["retrieval_count"], 1)
        m["net_help_rate"] = m["help_count"] / rc
        m["hurt_rate"] = m["hurt_count"] / rc
        m["net_score"] = m["net_help_rate"] - m["hurt_rate"]

    return metrics


def filter_skills(
    skills: list[dict],
    metrics: dict[str, dict],
    *,
    min_net_score: float = 0.0,
    max_useless_retrieval: int = 5,
) -> tuple[list[dict], list[dict]]:
    """Filter skills based on metrics. Returns (kept, removed)."""
    kept = []
    removed = []

    for skill in skills:
        if skill.get("status") != "success":
            removed.append({"skill": skill, "reason": "extraction_failed"})
            continue

        qid = skill["question_id"]
        m = metrics.get(qid, {})

        if m.get("net_score", 0) < min_net_score:
            removed.append({"skill": skill, "reason": "negative_net_score", "net_score": m.get("net_score", 0)})
        elif m.get("net_help_rate", 0) == 0 and m.get("retrieval_count", 0) >= max_useless_retrieval:
            removed.append({"skill": skill, "reason": "high_freq_useless", "retrieval_count": m.get("retrieval_count", 0)})
        else:
            kept.append(skill)

    return kept, removed


def merge_similar_skills(
    skills: list[dict],
    metrics: dict[str, dict],
    embed_model_name: str = "BAAI/bge-small-en-v1.5",
    similarity_threshold: float = 0.85,
) -> list[dict]:
    """Cluster similar skills by procedure embedding, keep best per cluster."""
    if len(skills) <= 1:
        return skills

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embed_model_name, device="cpu")
    procedures = [s.get("procedure", "") for s in skills]
    embeddings = model.encode(procedures, normalize_embeddings=True, show_progress_bar=False).astype("float32")

    # Greedy clustering: sorted by net_score descending
    sorted_indices = sorted(
        range(len(skills)),
        key=lambda i: metrics.get(skills[i]["question_id"], {}).get("net_score", 0),
        reverse=True,
    )

    clusters: list[list[int]] = []
    assigned = set()

    for idx in sorted_indices:
        if idx in assigned:
            continue
        cluster = [idx]
        assigned.add(idx)

        for other_idx in sorted_indices:
            if other_idx in assigned:
                continue
            sim = float(embeddings[idx] @ embeddings[other_idx])
            if sim >= similarity_threshold:
                cluster.append(other_idx)
                assigned.add(other_idx)

        clusters.append(cluster)

    # For each cluster, keep the representative (highest net_score) and merge keywords
    merged = []
    for cluster in clusters:
        rep_idx = cluster[0]
        rep_skill = dict(skills[rep_idx])

        if len(cluster) > 1:
            all_keywords = set()
            for idx in cluster:
                kw = skills[idx].get("keywords", "")
                if kw:
                    all_keywords.update(k.strip() for k in kw.split(",") if k.strip())
            rep_skill["keywords"] = ", ".join(sorted(all_keywords))
            rep_skill["_merged_from"] = [skills[i]["question_id"] for i in cluster]
            rep_skill["_cluster_size"] = len(cluster)

        merged.append(rep_skill)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and merge skills based on LOO metrics.")
    parser.add_argument("skill_file", help="Skills JSONL")
    parser.add_argument("loo_file", help="LOO predictions JSONL")
    parser.add_argument("baseline_file", help="Baseline (no-think) predictions JSONL")
    parser.add_argument("output_file", help="Output filtered skills JSONL")
    parser.add_argument("--report", help="Output filter report JSON")
    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--min-net-score", type=float, default=0.0)
    parser.add_argument("--max-useless-retrieval", type=int, default=5)
    parser.add_argument("--no-merge", action="store_true", help="Skip embedding-based merging")
    args = parser.parse_args()

    skills = load_jsonl(args.skill_file)
    loo_preds = load_jsonl(args.loo_file)
    baseline_preds = load_jsonl(args.baseline_file)

    print(f"Loaded {len(skills)} skills, {len(loo_preds)} LOO predictions, {len(baseline_preds)} baseline predictions")

    # Step 1: Compute metrics
    metrics = compute_skill_metrics(skills, loo_preds, baseline_preds)
    success_skills = [s for s in skills if s.get("status") == "success"]
    print(f"Success skills: {len(success_skills)}")

    # Print top/bottom skills by net_score
    ranked = sorted(metrics.items(), key=lambda x: x[1]["net_score"], reverse=True)
    print("\nTop 5 skills by NetScore:")
    for qid, m in ranked[:5]:
        print(f"  {qid}: NetScore={m['net_score']:.3f} (help={m['help_count']}, hurt={m['hurt_count']}, retrieved={m['retrieval_count']})")
    print("Bottom 5 skills by NetScore:")
    for qid, m in ranked[-5:]:
        print(f"  {qid}: NetScore={m['net_score']:.3f} (help={m['help_count']}, hurt={m['hurt_count']}, retrieved={m['retrieval_count']})")

    # Step 2: Filter
    kept, removed = filter_skills(
        skills, metrics,
        min_net_score=args.min_net_score,
        max_useless_retrieval=args.max_useless_retrieval,
    )
    print(f"\nAfter filtering: {len(kept)} kept, {len(removed)} removed")
    reason_counts = Counter(r["reason"] for r in removed)
    for reason, count in reason_counts.items():
        print(f"  Removed ({reason}): {count}")

    # Step 3: Merge
    if not args.no_merge and len(kept) > 1:
        print(f"\nMerging similar skills (threshold={args.similarity_threshold})...")
        merged = merge_similar_skills(
            kept, metrics,
            embed_model_name=args.embed_model,
            similarity_threshold=args.similarity_threshold,
        )
        n_clusters_merged = sum(1 for s in merged if s.get("_cluster_size", 1) > 1)
        print(f"After merging: {len(merged)} skills ({n_clusters_merged} clusters merged)")
    else:
        merged = kept

    # Write output
    write_jsonl(args.output_file, merged)
    print(f"\nWrote {len(merged)} filtered skills -> {args.output_file}")

    # Write report
    if args.report:
        report = {
            "original_count": len(skills),
            "success_count": len(success_skills),
            "after_filter": len(kept),
            "after_merge": len(merged),
            "removed_count": len(removed),
            "removal_reasons": dict(reason_counts),
            "metrics": {qid: m for qid, m in metrics.items()},
            "removed_skills": [{"question_id": r["skill"]["question_id"], "reason": r["reason"]} for r in removed],
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Wrote report -> {args.report}")


if __name__ == "__main__":
    main()
