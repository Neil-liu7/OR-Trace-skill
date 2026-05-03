"""Data I/O, text processing, and answer utilities."""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if "question_id" not in row:
                row["question_id"] = row.get("id", f"q_{idx}")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_prompt(
    template: str,
    *,
    problem: str = "",
    trace: str = "",
    answer: str = "",
    hints: str = "",
    examples: str = "",
) -> str:
    replacements = {
        "[INSERT PROBLEM HERE]": problem,
        "[INSERT CHAIN OF THOUGHT HERE]": trace,
        "[INSERT CORRECT ANSWER HERE]": answer,
        "{PROBLEM}": problem,
        "{SOLVING_HINTS}": hints,
        "{TRACE}": trace,
        "{ANSWER}": answer,
        "{EXAMPLES}": examples,
    }
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


# ---------------------------------------------------------------------------
# XML extraction
# ---------------------------------------------------------------------------

def extract_xml_content(text: str, tag: str) -> str:
    match = re.search(
        rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Answer normalization and matching
# ---------------------------------------------------------------------------

def find_last_boxed(text: str) -> str:
    marker = "\\boxed"
    idx = text.rfind(marker)
    if idx < 0:
        return ""
    brace_idx = text.find("{", idx)
    if brace_idx < 0:
        return ""
    depth = 0
    chars: list[str] = []
    for ch in text[brace_idx:]:
        if ch == "{":
            depth += 1
            if depth > 1:
                chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(ch)
        else:
            chars.append(ch)
    return ""


def extract_final_answer(text: str) -> str:
    boxed = find_last_boxed(text)
    if boxed:
        return boxed
    patterns = [
        r"(?im)^\s*final\s+answer\s*[:：]\s*(.+?)\s*$",
        r"(?im)^\s*answer\s*[:：]\s*(.+?)\s*$",
        r"(?is)\bthe\s+final\s+answer\s+is\s+(.+?)(?:\n|$)",
        r"(?is)\bthe\s+answer\s+is\s+(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            return matches[-1].group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\\boxed\s*{([^{}]*)}", r"\1", text)
    text = re.sub(r"\\text\s*{([^{}]*)}", r"\1", text)
    for token in ["$", "\\(", "\\)", "\\[", "\\]", "\\left", "\\right", "`", "*"]:
        text = text.replace(token, "")
    text = text.replace(" ", "")
    text = text.rstrip(".;,")
    return text


def answers_match(candidate: str, gold: str) -> bool:
    c = normalize_answer(candidate)
    g = normalize_answer(gold)
    if not c or not g:
        return False
    return c == g
