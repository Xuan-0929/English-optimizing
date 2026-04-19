#!/usr/bin/env python3
"""Build type-consistency DPO pairs and merge with existing DPO data.

Goal:
- Keep correction text unchanged.
- Teach model to prefer the correct `错误类型` label.

Input:
  - SFT train/val JSONL (instruction/input/output)
  - Existing DPO train/val JSONL (prompt/chosen/rejected)

Output:
  - type-consistency-only DPO files
  - merged DPO v2 files
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

TYPE_POOL = ["主谓一致", "时态一致", "介词", "定语从句"]


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


def choose_wrong_type(correct_type: str, rng: random.Random) -> str:
    candidates = [t for t in TYPE_POOL if t != correct_type]
    return rng.choice(candidates) if candidates else correct_type


def build_type_pairs_from_sft(
    sft_rows: List[Dict], sample_ratio: float, rng: random.Random
) -> List[Dict]:
    pairs: List[Dict] = []
    for row in sft_rows:
        instruction = str(row.get("instruction", "")).strip()
        user_input = str(row.get("input", "")).strip()
        output = str(row.get("output", "")).strip()
        if not instruction or not user_input or not output:
            continue
        parsed = parse_output(output)
        if parsed is None:
            continue

        if rng.random() > sample_ratio:
            continue

        error_type, correction, explanation = parsed
        wrong_type = choose_wrong_type(error_type, rng)
        chosen = format_output(error_type, correction, explanation)
        rejected = format_output(wrong_type, correction, explanation)
        if chosen == rejected:
            continue

        prompt = f"{instruction}\n{user_input}".strip()
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "pair_source": "type_consistency",
            }
        )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DPO type-consistency pairs.")
    parser.add_argument(
        "--sft-train",
        default="data/processed_v6/sft_train_v6_train.jsonl",
        help="SFT train JSONL path.",
    )
    parser.add_argument(
        "--sft-val",
        default="data/processed_v6/sft_train_v6_val.jsonl",
        help="SFT val JSONL path.",
    )
    parser.add_argument(
        "--dpo-train-base",
        default="data/processed_v6/dpo_train_v1.jsonl",
        help="Base DPO train JSONL path.",
    )
    parser.add_argument(
        "--dpo-val-base",
        default="data/processed_v6/dpo_val_v1.jsonl",
        help="Base DPO val JSONL path.",
    )
    parser.add_argument(
        "--type-train-out",
        default="data/processed_v6/dpo_type_train_v1.jsonl",
        help="Type-consistency DPO train output path.",
    )
    parser.add_argument(
        "--type-val-out",
        default="data/processed_v6/dpo_type_val_v1.jsonl",
        help="Type-consistency DPO val output path.",
    )
    parser.add_argument(
        "--merged-train-out",
        default="data/processed_v6/dpo_train_v2.jsonl",
        help="Merged DPO train output path.",
    )
    parser.add_argument(
        "--merged-val-out",
        default="data/processed_v6/dpo_val_v2.jsonl",
        help="Merged DPO val output path.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.5,
        help="Sampling ratio from SFT rows to build type-consistency pairs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.sample_ratio <= 1.0:
        raise ValueError("--sample-ratio must be in (0, 1].")

    rng = random.Random(args.seed)

    sft_train = read_jsonl(Path(args.sft_train))
    sft_val = read_jsonl(Path(args.sft_val))
    dpo_train_base = read_jsonl(Path(args.dpo_train_base))
    dpo_val_base = read_jsonl(Path(args.dpo_val_base))

    dpo_type_train = build_type_pairs_from_sft(sft_train, args.sample_ratio, rng)
    dpo_type_val = build_type_pairs_from_sft(sft_val, args.sample_ratio, rng)

    rng.shuffle(dpo_type_train)
    rng.shuffle(dpo_type_val)

    dpo_train_v2 = dpo_train_base + dpo_type_train
    dpo_val_v2 = dpo_val_base + dpo_type_val
    rng.shuffle(dpo_train_v2)
    rng.shuffle(dpo_val_v2)

    write_jsonl(Path(args.type_train_out), dpo_type_train)
    write_jsonl(Path(args.type_val_out), dpo_type_val)
    write_jsonl(Path(args.merged_train_out), dpo_train_v2)
    write_jsonl(Path(args.merged_val_out), dpo_val_v2)

    print("===== DPO Type-Consistency Data Prepared =====")
    print(f"sft_train: {len(sft_train)} -> dpo_type_train: {len(dpo_type_train)}")
    print(f"sft_val: {len(sft_val)} -> dpo_type_val: {len(dpo_type_val)}")
    print(f"dpo_train_base: {len(dpo_train_base)} -> dpo_train_v2: {len(dpo_train_v2)}")
    print(f"dpo_val_base: {len(dpo_val_base)} -> dpo_val_v2: {len(dpo_val_v2)}")
    print(f"wrote: {args.type_train_out}")
    print(f"wrote: {args.type_val_out}")
    print(f"wrote: {args.merged_train_out}")
    print(f"wrote: {args.merged_val_out}")


if __name__ == "__main__":
    main()

