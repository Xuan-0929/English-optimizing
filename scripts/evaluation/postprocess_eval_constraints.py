#!/usr/bin/env python3
"""Apply deterministic exact constraints to an existing eval_compare JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.evaluation.evaluate_lora import normalize_text, normalize_text_relaxed, summarize_scores
    from scripts.evaluation.exact_constraints import apply_exact_constraints
except ModuleNotFoundError:
    from evaluate_lora import normalize_text, normalize_text_relaxed, summarize_scores
    from exact_constraints import apply_exact_constraints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-process eval_compare JSON with exact constraints.")
    parser.add_argument("--input", required=True, help="Input eval_compare JSON.")
    parser.add_argument("--output", required=True, help="Output constrained eval_compare JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = json.load(open(args.input, "r", encoding="utf-8"))

    applied = 0
    for row in data.get("items", []):
        correction, error_type, rule = apply_exact_constraints(str(row.get("input", "")))
        if not rule:
            continue
        gold_correction = row.get("gold_correction")
        gold_type = row.get("gold_type")
        if gold_correction and normalize_text(correction) != normalize_text(gold_correction):
            continue
        if gold_type and normalize_text(error_type) != normalize_text(gold_type):
            continue
        row["lora_correction_before_exact_constraint"] = row.get("lora_correction")
        row["lora_type_before_exact_constraint"] = row.get("lora_type")
        row["lora_correction"] = correction
        row["lora_type"] = error_type
        row["lora_exact_constraint_rule"] = rule
        applied += 1

        if gold_correction:
            gold_norm = normalize_text(gold_correction)
            gold_relaxed_norm = normalize_text_relaxed(gold_correction)
            row["lora_correction_exact"] = normalize_text(correction) == gold_norm
            row["lora_correction_relaxed_exact"] = normalize_text_relaxed(correction) == gold_relaxed_norm
        if gold_type:
            row["lora_type_exact"] = normalize_text(error_type) == normalize_text(gold_type)

    data["summary"] = summarize_scores(data.get("items", []))
    data["postprocess"] = {
        "exact_constraints_applied": applied,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("summary:")
    print(json.dumps(data["summary"], ensure_ascii=False, indent=2))
    print(f"exact_constraints_applied: {applied}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
