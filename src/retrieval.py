"""BM25 + Embedding hybrid retrieval engine and skill bank."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import numpy as np


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


class BM25:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.tokenized = [tokenize(doc) for doc in documents]
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in self.tokenized]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.idf: dict[str, float] = {}
        self.freqs: list[dict[str, int]] = []
        self._index()

    def _index(self) -> None:
        doc_counts: dict[str, int] = {}
        for doc in self.tokenized:
            freqs: dict[str, int] = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            self.freqs.append(freqs)
            for token in freqs:
                doc_counts[token] = doc_counts.get(token, 0) + 1
        n_docs = max(len(self.tokenized), 1)
        for token, count in doc_counts.items():
            self.idf[token] = math.log(1 + (n_docs - count + 0.5) / (count + 0.5))

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        query_tokens = tokenize(query)
        scored: list[tuple[int, float]] = []
        for idx, freqs in enumerate(self.freqs):
            score = 0.0
            doc_len = self.doc_len[idx] or 1
            for token in query_tokens:
                freq = freqs.get(token, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                score += self.idf.get(token, 0.0) * (freq * (self.k1 + 1) / max(denom, 1e-9))
            if score > 0:
                scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def build_skill_document(row: dict[str, Any]) -> str:
    parts = [
        row.get("question", ""),
        row.get("procedure", ""),
        row.get("worked_example", ""),
        row.get("inject_text", ""),
        row.get("keywords", ""),
    ]
    return "\n".join(str(p).strip() for p in parts if str(p).strip())


def preferred_skill_text(row: dict[str, Any]) -> str:
    return row.get("inject_text") or row.get("procedure") or ""


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------

def rrf_fuse(rank_lists: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion across multiple ranked index lists."""
    score: dict[int, float] = {}
    for ranks in rank_lists:
        for r, doc_idx in enumerate(ranks):
            score[doc_idx] = score.get(doc_idx, 0.0) + 1.0 / (k + r + 1)
    fused = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [doc_idx for doc_idx, _ in fused]


# ---------------------------------------------------------------------------
# Structural signature for LP problems
# ---------------------------------------------------------------------------

def question_structure(question: str) -> str:
    q = question.lower()
    obj = "min" if "minim" in q else ("max" if "maxim" in q else "?")

    n_upper = len(re.findall(
        r"cannot exceed|at most|no more than|shall not|does not exceed|"
        r"not.*greater|upper|maximum.*(?:is|of)\s+\d",
        q,
    ))
    n_lower = len(re.findall(
        r"at least|no less|must.*at least|not.*less than|"
        r"minimum.*(?:is|of)\s+\d",
        q,
    ))

    nums = re.findall(r"\b\d+\.?\d*\b", question)
    n_bucket = "lo" if len(nums) <= 5 else ("mid" if len(nums) <= 9 else "hi")

    return f"{obj}|up{n_upper}|lo{n_lower}|n{n_bucket}"


class SkillBank:
    def __init__(self, records: list[dict[str, Any]], embed_model: str | None = None) -> None:
        self.records = [r for r in records if preferred_skill_text(r).strip()]
        documents = [build_skill_document(r) for r in self.records]
        self.bm25 = BM25(documents)

        self._struct_index: dict[str, list[int]] = defaultdict(list)
        for idx, rec in enumerate(self.records):
            sig = question_structure(rec.get("question", ""))
            self._struct_index[sig].append(idx)

        self._embed_model = None
        self.doc_vecs: np.ndarray | None = None
        if embed_model:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(embed_model, device="cpu")
            self.doc_vecs = self._embed_model.encode(
                documents, normalize_embeddings=True, show_progress_bar=True,
            ).astype("float32")

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        hits = []
        for idx, score in self.bm25.top_k(query, top_k):
            row = self.records[idx]
            hits.append({
                "record": row,
                "score": score,
                "question_id": row.get("question_id", ""),
                "skill_text": preferred_skill_text(row),
            })
        return hits

    def structural_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        sig = question_structure(query)
        candidate_idxs = self._struct_index.get(sig, [])

        if len(candidate_idxs) < 2:
            return self.search(query, top_k)

        sub_records = [self.records[i] for i in candidate_idxs]
        sub_docs = [build_skill_document(r) for r in sub_records]
        sub_bm25 = BM25(sub_docs)

        hits = []
        for sub_idx, score in sub_bm25.top_k(query, top_k):
            row = sub_records[sub_idx]
            hits.append({
                "record": row,
                "score": score,
                "question_id": row.get("question_id", ""),
                "skill_text": preferred_skill_text(row),
            })
        return hits

    def adaptive_search(
        self,
        query: str,
        top_k: int,
        *,
        min_score: float = 40.0,
        top1_ratio: float = 1.5,
        confidence_threshold: float = 50.0,
        ratio_cutoff: float = 0.7,
    ) -> list[dict[str, Any]]:
        raw_hits = self.search(query, top_k)
        if not raw_hits:
            return []

        hits = [h for h in raw_hits if h["score"] >= min_score]
        if not hits:
            return []

        top1_score = hits[0]["score"]

        if top1_score < confidence_threshold:
            return []

        if len(hits) >= 2 and top1_score / hits[1]["score"] >= top1_ratio:
            return hits[:1]

        cutoff = top1_score * ratio_cutoff
        return [h for h in hits if h["score"] >= cutoff]

    def embed_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self._embed_model is None or self.doc_vecs is None:
            return []
        q_vec = self._embed_model.encode([query], normalize_embeddings=True).astype("float32")[0]
        scores = self.doc_vecs @ q_vec
        top_idxs = np.argsort(-scores)[:top_k]
        hits = []
        for idx in top_idxs:
            idx = int(idx)
            if scores[idx] <= 0:
                break
            row = self.records[idx]
            hits.append({
                "record": row,
                "score": float(scores[idx]),
                "question_id": row.get("question_id", ""),
                "skill_text": preferred_skill_text(row),
            })
        return hits

    def hybrid_search(self, query: str, top_k: int, *, rrf_k: int = 60, n_candidates: int = 50) -> list[dict[str, Any]]:
        bm25_ranked = [idx for idx, _ in self.bm25.top_k(query, n_candidates)]
        if self._embed_model is not None and self.doc_vecs is not None:
            q_vec = self._embed_model.encode([query], normalize_embeddings=True).astype("float32")[0]
            scores = self.doc_vecs @ q_vec
            embed_ranked = [int(i) for i in np.argsort(-scores)[:n_candidates] if scores[i] > 0]
            fused_idxs = rrf_fuse([bm25_ranked, embed_ranked], k=rrf_k)
        else:
            fused_idxs = bm25_ranked

        hits = []
        for rank, idx in enumerate(fused_idxs[:top_k]):
            row = self.records[idx]
            hits.append({
                "record": row,
                "score": 1.0 / (rank + 1),
                "question_id": row.get("question_id", ""),
                "skill_text": preferred_skill_text(row),
            })
        return hits

    def structural_hybrid_search(self, query: str, top_k: int, *, rrf_k: int = 60) -> list[dict[str, Any]]:
        """Structural pre-filter → BM25 + Embedding RRF on the filtered subset."""
        sig = question_structure(query)
        candidate_idxs = self._struct_index.get(sig, [])

        if len(candidate_idxs) < 2:
            return self.hybrid_search(query, top_k, rrf_k=rrf_k)

        sub_records = [self.records[i] for i in candidate_idxs]
        sub_docs = [build_skill_document(r) for r in sub_records]
        sub_bm25 = BM25(sub_docs)

        n_candidates = len(candidate_idxs)
        bm25_ranked = [sub_idx for sub_idx, _ in sub_bm25.top_k(query, n_candidates)]

        if self._embed_model is not None and self.doc_vecs is not None:
            q_vec = self._embed_model.encode([query], normalize_embeddings=True).astype("float32")[0]
            sub_vecs = self.doc_vecs[candidate_idxs]
            scores = sub_vecs @ q_vec
            embed_ranked = [int(i) for i in np.argsort(-scores)[:n_candidates] if scores[i] > 0]
            fused_sub_idxs = rrf_fuse([bm25_ranked, embed_ranked], k=rrf_k)
        else:
            fused_sub_idxs = bm25_ranked

        hits = []
        for rank, sub_idx in enumerate(fused_sub_idxs[:top_k]):
            row = sub_records[sub_idx]
            hits.append({
                "record": row,
                "score": 1.0 / (rank + 1),
                "question_id": row.get("question_id", ""),
                "skill_text": preferred_skill_text(row),
            })
        return hits
