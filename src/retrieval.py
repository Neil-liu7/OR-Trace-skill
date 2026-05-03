"""BM25 retrieval engine and skill bank."""
from __future__ import annotations

import math
import re
from typing import Any


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


class SkillBank:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = [r for r in records if preferred_skill_text(r).strip()]
        documents = [build_skill_document(r) for r in self.records]
        self.bm25 = BM25(documents)

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
