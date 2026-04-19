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
from pathlib import Path
from typing import Dict, List


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


def make_rejected_identity(user_input: str) -> str:
    """Deliberately weak response used as rejected sample."""
    return (
        "**错误类型**: 时态一致\n"
        f"**改正**: {user_input}\n"
        "**解释**: 句子已正确，无需修改。"
    )


def build_pairs(rows: List[Dict]) -> List[Dict]:
    pairs: List[Dict] = []
    for row in rows:
        instruction = str(row.get("instruction", "")).strip()
        user_input = str(row.get("input", "")).strip()
        chosen = str(row.get("output", "")).strip()
        if not instruction or not user_input or not chosen:
            continue
        prompt = f"{instruction}\n{user_input}".strip()
        rejected = make_rejected_identity(user_input)
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

    train_rows = read_jsonl(Path(args.input_train))
    val_rows = read_jsonl(Path(args.input_val))

    dpo_train = build_pairs(train_rows)
    dpo_val = build_pairs(val_rows)

    random.Random(args.seed).shuffle(dpo_train)
    random.Random(args.seed).shuffle(dpo_val)

    write_jsonl(Path(args.output_train), dpo_train)
    write_jsonl(Path(args.output_val), dpo_val)

    print("===== DPO Data Prepared =====")
    print(f"sft_train: {len(train_rows)} -> dpo_train: {len(dpo_train)}")
    print(f"sft_val: {len(val_rows)} -> dpo_val: {len(dpo_val)}")
    print(f"wrote: {args.output_train}")
    print(f"wrote: {args.output_val}")


if __name__ == "__main__":
    main()

