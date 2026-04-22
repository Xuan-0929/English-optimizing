#!/usr/bin/env python3
"""Build DPO preference pairs from SFT-style JSONL.

Input row format:
  {"instruction": "...", "input": "...", "output": "..."}

Output row format:
  {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TYPE_RE = re.compile(r"\*\*错误类型\*\*:\s*(.+?)\n")
CORR_RE = re.compile(r"\*\*改正\*\*:\s*(.+?)\n", re.S)
TYPE_POOL = ["主谓一致", "时态一致", "介词", "定语从句"]


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


def parse_output_type_and_corr(text: str) -> Optional[Tuple[str, str]]:
    m_type = TYPE_RE.search(text)
    m_corr = CORR_RE.search(text)
    if not (m_type and m_corr):
        return None
    return m_type.group(1).strip(), m_corr.group(1).strip()


def format_output(error_type: str, correction: str, explanation: str) -> str:
    return (
        f"**错误类型**: {error_type}\n"
        f"**改正**: {correction}\n"
        f"**解释**: {explanation}"
    )


def choose_wrong_type(correct_type: str, rng: random.Random, type_counter: Dict[str, int]) -> str:
    candidates = [t for t in TYPE_POOL if t != correct_type]
    if not candidates:
        return "时态一致"
    # Prefer under-represented wrong types to avoid long-term type skew.
    min_count = min(type_counter.get(t, 0) for t in candidates)
    least_used = [t for t in candidates if type_counter.get(t, 0) == min_count]
    return rng.choice(least_used)


def make_rejected_from_row(
    user_input: str,
    chosen: str,
    rng: random.Random,
    type_counter: Dict[str, int],
) -> str:
    parsed = parse_output_type_and_corr(chosen)
    if parsed is None:
        rejected_type = "时态一致"
    else:
        correct_type, _ = parsed
        rejected_type = choose_wrong_type(correct_type, rng, type_counter)
    type_counter[rejected_type] = type_counter.get(rejected_type, 0) + 1
    return format_output(
        rejected_type,
        user_input,
        "该改法未修正关键语法问题，句子仍有错误。",
    )


def build_pairs(rows: List[Dict], rng: random.Random) -> List[Dict]:
    pairs: List[Dict] = []
    rejected_type_counter: Dict[str, int] = {}
    for row in rows:
        instruction = str(row.get("instruction", "")).strip()
        user_input = str(row.get("input", "")).strip()
        chosen = str(row.get("output", "")).strip()
        if not instruction or not user_input or not chosen:
            continue
        prompt = f"{instruction}\n{user_input}".strip()
        rejected = make_rejected_from_row(user_input, chosen, rng, rejected_type_counter)
        if rejected == chosen:
            # Skip degenerate rows.
            continue
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    if rejected_type_counter:
        print("rejected_type_distribution:", json.dumps(rejected_type_counter, ensure_ascii=False))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DPO preference dataset.")
    parser.add_argument(
        "--input-train",
        default="data/processed_v6/sft_train_v6_train.jsonl",
        help="SFT train JSONL path.",
    )
    parser.add_argument(
        "--input-val",
        default="data/processed_v6/sft_train_v6_val.jsonl",
        help="SFT val JSONL path.",
    )
    parser.add_argument(
        "--output-train",
        default="data/processed_v6/dpo_train_v1.jsonl",
        help="DPO train JSONL output path.",
    )
    parser.add_argument(
        "--output-val",
        default="data/processed_v6/dpo_val_v1.jsonl",
        help="DPO val JSONL output path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    train_rows = read_jsonl(Path(args.input_train))
    val_rows = read_jsonl(Path(args.input_val))

    dpo_train = build_pairs(train_rows, rng)
    dpo_val = build_pairs(val_rows, rng)

    rng.shuffle(dpo_train)
    rng.shuffle(dpo_val)

    write_jsonl(Path(args.output_train), dpo_train)
    write_jsonl(Path(args.output_val), dpo_val)

    print("===== DPO Data Prepared =====")
    print(f"sft_train: {len(train_rows)} -> dpo_train: {len(dpo_train)}")
    print(f"sft_val: {len(val_rows)} -> dpo_val: {len(dpo_val)}")
    print(f"wrote: {args.output_train}")
    print(f"wrote: {args.output_val}")


if __name__ == "__main__":
    main()
