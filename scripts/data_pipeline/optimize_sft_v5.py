#!/usr/bin/env python3
"""Per-row optimization pass for v4 dataset.

Goal:
- review every row
- improve correction text for known bad collocations
- rewrite explanation to align with actual edit
- drop rows that remain high-risk after optimization
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TYPE_RE = re.compile(r"\*\*错误类型\*\*:\s*(.+?)\n")
CORR_RE = re.compile(r"\*\*改正\*\*:\s*(.+?)\n", re.S)
EXP_RE = re.compile(r"\*\*解释\*\*:\s*(.+)$", re.S)
WORD_RE = re.compile(r"[A-Za-z']+")
CN_RE = re.compile(r"[\u4e00-\u9fff]")


def parse_output(output: str) -> Optional[Tuple[str, str, str]]:
    m_type = TYPE_RE.search(output)
    m_corr = CORR_RE.search(output)
    m_exp = EXP_RE.search(output)
    if not (m_type and m_corr and m_exp):
        return None
    return m_type.group(1).strip(), m_corr.group(1).strip(), m_exp.group(1).strip()


def normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fix_known_correction_patterns(inp: str, corr: str, error_type: str) -> Tuple[str, List[str]]:
    """Apply conservative and high-confidence fixes only."""
    actions: List[str] = []
    c = corr

    # Common collocation cleanup
    rules = [
        (r"\bin English language\b", "in English", "fix_in_english_language"),
        (r"\blisten(?:ing)? at the music\b", "listen to music", "fix_listen_at_music"),
        (r"\bwatch(?:es|ed|ing)? at the movie\b", "watch the movie", "fix_watch_at_movie"),
        (r"\bread(?:s|ing)? at the book\b", "read the book", "fix_read_at_book"),
        (r"\bsurf(?:s|ed|ing)? at the internet\b", "surf the internet", "fix_surf_at_internet"),
        (r"\bplay at the board game\b", "play board games", "fix_play_at_board_game"),
        (r"\bwalk at the beach\b", "walk on the beach", "fix_walk_at_beach"),
        (r"\bwrite(?:s|ing| wrote)? at the computer\b", "write on the computer", "fix_write_at_computer"),
        (r"\bclean(?:s|ed|ing)? at the house\b", "clean the house", "fix_clean_at_house"),
    ]
    for pat, rep, tag in rules:
        new = re.sub(pat, rep, c, flags=re.I)
        if new != c:
            c = new
            actions.append(tag)

    # Keep sentence punctuation consistent.
    c = normalize_spaces(c)
    if c and c[-1] not in ".!?":
        c += "."
        actions.append("ensure_trailing_punctuation")

    return c, actions


def diff_hint(inp: str, corr: str) -> str:
    """Get a short edit hint like 'is -> are' if possible."""
    a = WORD_RE.findall(inp)
    b = WORD_RE.findall(corr)
    sm = difflib.SequenceMatcher(None, [x.lower() for x in a], [x.lower() for x in b])
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in {"replace", "delete", "insert"}:
            left = " ".join(a[i1:i2]).strip()
            right = " ".join(b[j1:j2]).strip()
            if left or right:
                left = left or "∅"
                right = right or "∅"
                return f"{left} -> {right}"
    return "局部词形或搭配已调整"


def make_explanation(error_type: str, inp: str, corr: str) -> str:
    hint = diff_hint(inp, corr)
    if error_type == "主谓一致":
        return f"主语和谓语需要在人称或单复数上保持一致，本句已将 `{hint}`。"
    if error_type == "时态一致":
        return f"根据主句语境与时间关系调整时态，本句已将 `{hint}`。"
    if error_type == "介词":
        return f"该处应使用固定搭配或更自然的介词结构，本句已将 `{hint}`。"
    if error_type == "定语从句":
        return f"定语从句中关系词或从句结构需与先行词匹配，本句已将 `{hint}`。"
    return f"已按语法规则调整本句，修改点：`{hint}`。"


def risky_after_opt(inp: str, corr: str, error_type: str) -> List[str]:
    reasons: List[str] = []
    if not corr:
        reasons.append("empty_correction")
        return reasons

    if CN_RE.search(corr):
        reasons.append("correction_contains_chinese")

    sim = difflib.SequenceMatcher(None, inp.lower(), corr.lower()).ratio()
    thresholds = {
        "主谓一致": 0.82,
        "时态一致": 0.80,
        "介词": 0.78,
        "定语从句": 0.80,
    }
    th = thresholds.get(error_type, 0.80)
    if sim < th:
        reasons.append(f"low_similarity_{sim:.2f}")

    # Still suspicious if these remain after fix.
    lower_corr = corr.lower()
    for bad in ["rather than", "preferred to", " at the music", " at the book", " at the movie", " at the internet"]:
        if bad in lower_corr:
            reasons.append(f"suspicious_phrase:{bad.strip()}")

    return reasons


def format_output(error_type: str, correction: str, explanation: str) -> str:
    return (
        f"**错误类型**: {error_type}\n"
        f"**改正**: {correction}\n"
        f"**解释**: {explanation}"
    )


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-row optimization for v4 dataset.")
    parser.add_argument("--input", default="data/processed_v4/sft_train_clean_v4.jsonl")
    parser.add_argument("--output-dir", default="data/processed_v5")
    parser.add_argument("--review-dir", default="data/review_pack")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    output_dir = Path(args.output_dir)
    review_dir = Path(args.review_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    kept: List[Dict] = []
    removed: List[Dict] = []
    audit: List[Dict] = []
    removed_reason_count = Counter()
    type_count_before = Counter()
    type_count_after = Counter()

    for idx, row in enumerate(rows, start=1):
        inp = str(row.get("input", "")).strip()
        out = str(row.get("output", ""))
        parsed = parse_output(out)

        if parsed is None:
            removed.append(
                {
                    "sample_id": f"V5R{len(removed)+1:04d}",
                    "source_index": idx,
                    "input": inp,
                    "current_output": out,
                    "remove_reasons": ["format_invalid"],
                }
            )
            removed_reason_count["format_invalid"] += 1
            continue

        error_type, corr, exp = parsed
        type_count_before[error_type] += 1

        opt_corr, actions = fix_known_correction_patterns(inp, corr, error_type)
        opt_exp = make_explanation(error_type, inp, opt_corr)
        risk = risky_after_opt(inp, opt_corr, error_type)

        audit.append(
            {
                "sample_id": f"V5A{idx:04d}",
                "source_index": idx,
                "error_type": error_type,
                "input": inp,
                "old_correction": corr,
                "new_correction": opt_corr,
                "old_explanation": exp,
                "new_explanation": opt_exp,
                "actions": actions,
                "risk_flags": risk,
            }
        )

        if risk:
            removed.append(
                {
                    "sample_id": f"V5R{len(removed)+1:04d}",
                    "source_index": idx,
                    "error_type": error_type,
                    "input": inp,
                    "current_output": out,
                    "proposed_output": format_output(error_type, opt_corr, opt_exp),
                    "remove_reasons": risk,
                }
            )
            for x in risk:
                removed_reason_count[x] += 1
            continue

        row["output"] = format_output(error_type, opt_corr, opt_exp)
        kept.append(row)
        type_count_after[error_type] += 1

    import random

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    split = max(1, int(len(kept) * (1 - args.val_ratio)))
    train_rows = kept[:split]
    val_rows = kept[split:]

    clean_out = output_dir / "sft_train_clean_v5.jsonl"
    train_out = output_dir / "sft_train_v5_train.jsonl"
    val_out = output_dir / "sft_train_v5_val.jsonl"
    write_jsonl(clean_out, kept)
    write_jsonl(train_out, train_rows)
    write_jsonl(val_out, val_rows)

    removed_jsonl = review_dir / "removed_high_risk_v5.jsonl"
    audit_jsonl = review_dir / "optimization_audit_v5.jsonl"
    write_jsonl(removed_jsonl, removed)
    write_jsonl(audit_jsonl, audit)

    removed_csv = review_dir / "removed_high_risk_v5.csv"
    with removed_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "source_index", "error_type", "input", "remove_reasons"])
        for r in removed:
            w.writerow(
                [
                    r.get("sample_id", ""),
                    r.get("source_index", ""),
                    r.get("error_type", ""),
                    r.get("input", ""),
                    "|".join(r.get("remove_reasons", [])),
                ]
            )

    report = {
        "input_file": str(args.input),
        "total_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "removed_ratio": round(len(removed) / max(1, len(rows)), 4),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "type_count_before": dict(type_count_before),
        "type_count_after": dict(type_count_after),
        "removed_reason_counts": dict(removed_reason_count),
        "outputs": {
            "clean_v5": str(clean_out),
            "train_v5": str(train_out),
            "val_v5": str(val_out),
            "removed_jsonl": str(removed_jsonl),
            "removed_csv": str(removed_csv),
            "audit_jsonl": str(audit_jsonl),
        },
    }
    report_path = review_dir / "cleaning_report_v5.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

