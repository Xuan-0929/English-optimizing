#!/usr/bin/env python3
"""Build a safer v2 SFT dataset by removing high-risk noisy samples.

The script does NOT attempt aggressive auto-rewrite. It removes rows that
match strong noise patterns and exports:
1) cleaned full/train/val JSONL files
2) a review pack for removed rows
3) a JSON report with rule-level counts
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


TYPE_RE = re.compile(r"\*\*错误类型\*\*:\s*(.+?)\n")
CORR_RE = re.compile(r"\*\*改正\*\*:\s*(.+?)\n", re.S)
EXP_RE = re.compile(r"\*\*解释\*\*:\s*(.+)$", re.S)
CN_RE = re.compile(r"[\u4e00-\u9fff]")
INV_MARKER_RE = re.compile(r"\bonly\b|\bnever\b|\brarely\b|\bnot until\b", re.I)


def text_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def parse_output(output: str) -> Tuple[str, str, str] | None:
    m_type = TYPE_RE.search(output)
    m_corr = CORR_RE.search(output)
    m_exp = EXP_RE.search(output)
    if not (m_type and m_corr and m_exp):
        return None
    return m_type.group(1).strip(), m_corr.group(1).strip(), m_exp.group(1).strip()


def detect_reasons(error_type: str, inp: str, corr: str, exp: str) -> List[str]:
    reasons: List[str] = []
    inp_l = inp.lower()
    corr_l = corr.lower()

    if "介词at错误地放在了时间之前，应放在时间之后" in exp:
        reasons.append("wrong_at_position_template")

    if error_type == "连接词":
        if ("rather than" in corr_l and "rather than" not in inp_l) or (
            "preferred to" in corr_l and "preferred to" not in inp_l
        ):
            reasons.append("connector_semantic_injection")

        if any(x in corr_l for x in ["although ", "despite ", "yet "]) and not any(
            x in inp_l for x in ["although ", "despite ", "yet "]
        ):
            reasons.append("connector_style_rewrite")

    if error_type == "倒装句":
        if INV_MARKER_RE.search(corr_l) and not INV_MARKER_RE.search(inp_l):
            reasons.append("forced_inversion")

    if error_type == "主谓一致" and any(
        x in corr_l for x in ["although ", "despite ", "rather than", "preferred to"]
    ):
        reasons.append("type_drift_sva_to_connector")

    if error_type in {"连接词", "介词", "倒装句"} and text_similarity(inp, corr) < 0.60:
        reasons.append("low_similarity_rewrite")

    if "tomorrow" in corr_l and re.search(r"\bvisited\b|\bhad\b", corr_l):
        reasons.append("tense_anomaly_tomorrow_past")

    if "mentioned" in inp_l and "had been" in inp_l and "has been" in corr_l:
        reasons.append("tense_backshift_inconsistent")

    if CN_RE.search(inp):
        reasons.append("input_contains_chinese")
    if CN_RE.search(corr):
        reasons.append("correction_contains_chinese")

    return reasons


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean high-risk noisy SFT rows.")
    parser.add_argument(
        "--input",
        default="data/processed/sft_train_clean.jsonl",
        help="Input clean JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed_v2",
        help="Output dir for v2 dataset files.",
    )
    parser.add_argument(
        "--review-pack-dir",
        default="data/review_pack",
        help="Output dir for removed sample review pack.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1).")

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    review_dir = Path(args.review_pack_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    kept: List[Dict] = []
    removed: List[Dict] = []
    reason_counter = Counter()
    reason_by_type = defaultdict(Counter)

    for idx, row in enumerate(rows, start=1):
        out = str(row.get("output", ""))
        parsed = parse_output(out)
        if parsed is None:
            reasons = ["format_invalid"]
            error_type = "UNKNOWN"
            corr = ""
            exp = ""
        else:
            error_type, corr, exp = parsed
            reasons = detect_reasons(error_type, str(row.get("input", "")), corr, exp)

        if reasons:
            removed.append(
                {
                    "sample_id": f"R{len(removed)+1:04d}",
                    "source_file": str(input_path),
                    "source_index": idx,
                    "error_type": error_type,
                    "input": row.get("input", ""),
                    "current_output": out,
                    "current_correction": corr,
                    "current_explanation": exp,
                    "remove_reasons": reasons,
                    "review_status": "todo",
                    "review_notes": "",
                    "revised_output": "",
                }
            )
            for reason in reasons:
                reason_counter[reason] += 1
                reason_by_type[reason][error_type] += 1
        else:
            kept.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    split_idx = max(1, int(len(kept) * (1 - args.val_ratio)))
    train_rows = kept[:split_idx]
    val_rows = kept[split_idx:]

    clean_out = out_dir / "sft_train_clean_v2.jsonl"
    train_out = out_dir / "sft_train_v2_train.jsonl"
    val_out = out_dir / "sft_train_v2_val.jsonl"
    write_jsonl(clean_out, kept)
    write_jsonl(train_out, train_rows)
    write_jsonl(val_out, val_rows)

    removed_jsonl = review_dir / "removed_high_risk_v2.jsonl"
    write_jsonl(removed_jsonl, removed)

    removed_csv = review_dir / "removed_high_risk_v2.csv"
    with removed_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "source_index",
                "error_type",
                "input",
                "current_correction",
                "remove_reasons",
                "review_status",
                "review_notes",
                "revised_output",
            ]
        )
        for item in removed:
            writer.writerow(
                [
                    item["sample_id"],
                    item["source_index"],
                    item["error_type"],
                    item["input"],
                    item["current_correction"],
                    "|".join(item["remove_reasons"]),
                    item["review_status"],
                    item["review_notes"],
                    item["revised_output"],
                ]
            )

    report = {
        "input_file": str(input_path),
        "total_rows": len(rows),
        "removed_rows": len(removed),
        "kept_rows": len(kept),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "removed_ratio": round(len(removed) / max(1, len(rows)), 4),
        "reason_counts": dict(reason_counter),
        "reason_counts_by_error_type": {
            k: dict(v) for k, v in reason_by_type.items()
        },
        "outputs": {
            "clean_v2": str(clean_out),
            "train_v2": str(train_out),
            "val_v2": str(val_out),
            "removed_jsonl": str(removed_jsonl),
            "removed_csv": str(removed_csv),
        },
    }
    report_path = review_dir / "cleaning_report_v2.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== v2 cleaning done =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
