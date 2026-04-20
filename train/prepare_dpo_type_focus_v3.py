#!/usr/bin/env python3
"""Prepare a small focused DPO v3 dataset for type-label consistency.

This script targets known confusion where correction is right but `错误类型` is wrong
(especially 介词 vs 时态一致).
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
EXP_RE = re.compile(r"\*\*解释\*\*:\s*(.+)$", re.S)


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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_output(output: str) -> Optional[Tuple[str, str, str]]:
    m_type = TYPE_RE.search(output)
    m_corr = CORR_RE.search(output)
    m_exp = EXP_RE.search(output)
    if not (m_type and m_corr and m_exp):
        return None
    return m_type.group(1).strip(), m_corr.group(1).strip(), m_exp.group(1).strip()


def format_output(error_type: str, correction: str, explanation: str) -> str:
    return (
        f"**错误类型**: {error_type}\n"
        f"**改正**: {correction}\n"
        f"**解释**: {explanation}"
    )


def should_focus(input_text: str, error_type: str) -> bool:
    if error_type != "介词":
        return False
    t = input_text.lower()
    markers = [
        "married with",
        "discussed about",
        "arrived to",
        "at next",
        "in hospital",
        "to school to meet",
        "familiar of",
        "agree with her at",
        "concur with him at",
        "for an hour at the",
    ]
    return any(m in t for m in markers)


def make_focus_pairs(
    sft_rows: List[Dict],
    rng: random.Random,
    max_pairs: int,
) -> List[Dict]:
    pairs: List[Dict] = []
    for row in sft_rows:
        instruction = str(row.get("instruction", "")).strip()
        user_input = str(row.get("input", "")).strip()
        output = str(row.get("output", "")).strip()
        parsed = parse_output(output)
        if not instruction or not user_input or parsed is None:
            continue
        et, corr, exp = parsed
        if not should_focus(user_input, et):
            continue

        prompt = f"{instruction}\n{user_input}".strip()
        chosen = format_output("介词", corr, exp)
        rejected = format_output("时态一致", corr, exp)
        if chosen == rejected:
            continue
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "pair_source": "type_focus_v3",
            }
        )

    rng.shuffle(pairs)
    return pairs[:max_pairs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare focused DPO v3 type pairs.")
    parser.add_argument("--sft-train", default="data/processed_v6/sft_train_v6_train.jsonl")
    parser.add_argument("--sft-val", default="data/processed_v6/sft_train_v6_val.jsonl")
    parser.add_argument("--dpo-train-base", default="data/processed_v6/dpo_train_v2.jsonl")
    parser.add_argument("--dpo-val-base", default="data/processed_v6/dpo_val_v2.jsonl")
    parser.add_argument("--focus-train-out", default="data/processed_v6/dpo_type_focus_train_v3.jsonl")
    parser.add_argument("--focus-val-out", default="data/processed_v6/dpo_type_focus_val_v3.jsonl")
    parser.add_argument("--merged-train-out", default="data/processed_v6/dpo_train_v3.jsonl")
    parser.add_argument("--merged-val-out", default="data/processed_v6/dpo_val_v3.jsonl")
    parser.add_argument("--max-train-pairs", type=int, default=60)
    parser.add_argument("--max-val-pairs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sft_train = read_jsonl(Path(args.sft_train))
    sft_val = read_jsonl(Path(args.sft_val))
    dpo_train_base = read_jsonl(Path(args.dpo_train_base))
    dpo_val_base = read_jsonl(Path(args.dpo_val_base))

    focus_train = make_focus_pairs(sft_train, rng, args.max_train_pairs)
    focus_val = make_focus_pairs(sft_val, rng, args.max_val_pairs)

    dpo_train_v3 = dpo_train_base + focus_train
    dpo_val_v3 = dpo_val_base + focus_val
    rng.shuffle(dpo_train_v3)
    rng.shuffle(dpo_val_v3)

    write_jsonl(Path(args.focus_train_out), focus_train)
    write_jsonl(Path(args.focus_val_out), focus_val)
    write_jsonl(Path(args.merged_train_out), dpo_train_v3)
    write_jsonl(Path(args.merged_val_out), dpo_val_v3)

    print("===== DPO Type Focus v3 Prepared =====")
    print(f"focus_train: {len(focus_train)}")
    print(f"focus_val: {len(focus_val)}")
    print(f"dpo_train_base: {len(dpo_train_base)} -> dpo_train_v3: {len(dpo_train_v3)}")
    print(f"dpo_val_base: {len(dpo_val_base)} -> dpo_val_v3: {len(dpo_val_v3)}")
    print(f"wrote: {args.focus_train_out}")
    print(f"wrote: {args.focus_val_out}")
    print(f"wrote: {args.merged_train_out}")
    print(f"wrote: {args.merged_val_out}")


if __name__ == "__main__":
    main()

