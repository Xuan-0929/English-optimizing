#!/usr/bin/env python3
"""Prepare clean train/val JSONL files for SFT."""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ERROR_TYPE_RE = re.compile(r"\*\*错误类型\*\*:\s*(.+?)\n")
CORRECTION_RE = re.compile(r"\*\*改正\*\*:\s*(.+?)\n")
EXPLANATION_RE = re.compile(r"\*\*解释\*\*:\s*(.+)$", re.DOTALL)
BE_VERB_AGREEMENT_RE = re.compile(
    r"\b(they|we|you)\s+is\b|\b(he|she|it)\s+are\b",
    re.IGNORECASE,
)
DO_VERB_AGREEMENT_RE = re.compile(
    r"\b(he|she|it)\s+don't\b|\b(i|you|we|they)\s+doesn't\b",
    re.IGNORECASE,
)
THIRD_PERSON_RE = re.compile(
    r"\b(he|she|it|my sister|my brother|the student|the teacher)\s+\w+\s+every day\b",
    re.IGNORECASE,
)
RELATIVE_CLAUSE_RE = re.compile(
    r"\b(which|that|who)\b.*\bit\b|\bthe person which\b|\bthe book who\b",
    re.IGNORECASE,
)


def parse_output(output: str) -> Optional[Tuple[str, str, str]]:
    match_type = ERROR_TYPE_RE.search(output)
    match_corr = CORRECTION_RE.search(output)
    match_exp = EXPLANATION_RE.search(output)
    if not (match_type and match_corr and match_exp):
        return None
    return (
        match_type.group(1).strip(),
        match_corr.group(1).strip(),
        match_exp.group(1).strip(),
    )


def format_output(error_type: str, correction: str, explanation: str) -> str:
    return (
        f"**错误类型**: {error_type}\n"
        f"**改正**: {correction}\n"
        f"**解释**: {explanation}"
    )


def normalize_error_type(user_input: str, correction: str, error_type: str) -> str:
    src = user_input.strip()
    tgt = correction.strip()
    if not src or not tgt:
        return error_type

    src_l = src.lower()
    tgt_l = tgt.lower()
    if src_l == tgt_l:
        return error_type

    if BE_VERB_AGREEMENT_RE.search(src) or DO_VERB_AGREEMENT_RE.search(src):
        return "主谓一致"

    if THIRD_PERSON_RE.search(src) and re.search(r"\b(he|she|it)\s+\w+s\b", tgt_l):
        return "主谓一致"

    if RELATIVE_CLAUSE_RE.search(src):
        return "定语从句"

    if (
        ("yesterday" in src_l or "last week" in src_l or "tomorrow" in src_l)
        and src_l != tgt_l
        and ("have been" in src_l or "had been" in src_l or "will" in src_l)
    ):
        return "时态一致"

    return error_type


def clean_records(
    records: List[Dict],
    min_explanation_chars: int = 8,
    relabel_rules: bool = True,
) -> Tuple[List[Dict], Dict[str, int], Counter]:
    cleaned: List[Dict] = []
    seen_inputs = set()
    stats = {
        "invalid_json_shape": 0,
        "invalid_output_format": 0,
        "empty_text": 0,
        "same_input_and_correction": 0,
        "duplicate_input": 0,
        "short_explanation": 0,
        "relabel_by_rule": 0,
    }
    error_types = Counter()

    for item in records:
        if not isinstance(item, dict):
            stats["invalid_json_shape"] += 1
            continue

        instruction = str(item.get("instruction", "")).strip()
        user_input = str(item.get("input", "")).strip()
        output = str(item.get("output", "")).strip()
        if not instruction or not user_input or not output:
            stats["empty_text"] += 1
            continue

        parsed = parse_output(output)
        if parsed is None:
            stats["invalid_output_format"] += 1
            continue
        error_type, correction, explanation = parsed

        if user_input.lower() == correction.lower():
            stats["same_input_and_correction"] += 1
            continue
        if len(explanation) < min_explanation_chars:
            stats["short_explanation"] += 1
            continue

        dedup_key = user_input.lower()
        if dedup_key in seen_inputs:
            stats["duplicate_input"] += 1
            continue
        seen_inputs.add(dedup_key)

        final_error_type = error_type
        if relabel_rules:
            fixed_error_type = normalize_error_type(user_input, correction, error_type)
            if fixed_error_type != error_type:
                stats["relabel_by_rule"] += 1
                final_error_type = fixed_error_type

        output = format_output(final_error_type, correction, explanation)

        cleaned.append(
            {
                "instruction": instruction,
                "input": user_input,
                "output": output,
            }
        )
        error_types[final_error_type] += 1

    return cleaned, stats, error_types


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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and split SFT JSONL data.")
    parser.add_argument(
        "--input",
        default="data/sft_train.jsonl",
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for clean/train/val files.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation ratio in (0, 1).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--min-explanation-chars",
        type=int,
        default=8,
        help="Drop records with too-short explanation.",
    )
    parser.add_argument(
        "--disable-relabel-rules",
        action="store_true",
        help="Disable rule-based error_type normalization.",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1).")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = read_jsonl(input_path)
    cleaned, stats, error_types = clean_records(
        all_records,
        min_explanation_chars=args.min_explanation_chars,
        relabel_rules=not args.disable_relabel_rules,
    )

    rng = random.Random(args.seed)
    rng.shuffle(cleaned)

    split_idx = max(1, int(len(cleaned) * (1 - args.val_ratio)))
    train_rows = cleaned[:split_idx]
    val_rows = cleaned[split_idx:]

    clean_path = output_dir / "sft_train_clean.jsonl"
    train_path = output_dir / "sft_train_v1_train.jsonl"
    val_path = output_dir / "sft_train_v1_val.jsonl"

    write_jsonl(clean_path, cleaned)
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    print("===== SFT Data Preparation =====")
    print(f"input: {input_path}")
    print(f"raw records: {len(all_records)}")
    print(f"clean records: {len(cleaned)}")
    print(f"train records: {len(train_rows)}")
    print(f"val records: {len(val_rows)}")
    print("filter stats:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")
    print("error type distribution:")
    for k, v in error_types.most_common():
        print(f"  - {k}: {v}")
    print(f"wrote: {clean_path}")
    print(f"wrote: {train_path}")
    print(f"wrote: {val_path}")


if __name__ == "__main__":
    main()
