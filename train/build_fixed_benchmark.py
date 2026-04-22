#!/usr/bin/env python3
"""Build a fixed, stratified benchmark JSONL from SFT validation data.

Output format keeps SFT rows:
  {"instruction": "...", "input": "...", "output": "..."}
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TYPE_RE = re.compile(r"\*\*错误类型\*\*:\s*(.+?)\n")
TYPE_ORDER = ["介词", "主谓一致", "时态一致", "定语从句"]
DEFAULT_TARGET = {"介词": 18, "主谓一致": 14, "时态一致": 6, "定语从句": 22}
DEFAULT_REQUIRED_INPUTS = {
    "She married with him in 2018.",
    "Clinical researches has strict ethics requirements.",
}


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


def parse_type(output: str) -> Optional[str]:
    m = TYPE_RE.search(output or "")
    if not m:
        return None
    return m.group(1).strip()


def stratified_sample(
    rows: List[Dict],
    target_by_type: Dict[str, int],
    required_inputs: set[str],
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict]] = {t: [] for t in TYPE_ORDER}
    for row in rows:
        out = str(row.get("output", "")).strip()
        t = parse_type(out)
        if t in grouped:
            grouped[t].append(row)

    for t in TYPE_ORDER:
        rng.shuffle(grouped[t])

    picked: List[Dict] = []
    used_inputs: set[str] = set()
    required_left = set(required_inputs)

    for t in TYPE_ORDER:
        need = target_by_type.get(t, 0)
        pool = grouped[t]
        if need <= 0 or not pool:
            continue

        type_required = []
        type_optional = []
        for row in pool:
            inp = str(row.get("input", "")).strip()
            if inp in required_left:
                type_required.append(row)
            else:
                type_optional.append(row)

        for row in type_required:
            inp = str(row.get("input", "")).strip()
            if inp and inp not in used_inputs:
                picked.append(row)
                used_inputs.add(inp)
                required_left.discard(inp)
                if sum(1 for r in picked if parse_type(str(r.get("output", ""))) == t) >= need:
                    break

        current_t = sum(1 for r in picked if parse_type(str(r.get("output", ""))) == t)
        for row in type_optional:
            if current_t >= need:
                break
            inp = str(row.get("input", "")).strip()
            if not inp or inp in used_inputs:
                continue
            picked.append(row)
            used_inputs.add(inp)
            current_t += 1

    rng.shuffle(picked)
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed benchmark from SFT val data.")
    parser.add_argument("--input-val", default="data/processed_v6/sft_train_v6_val.jsonl")
    parser.add_argument("--output", default="data/processed_v6/sft_eval_benchmark_v1.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_val))
    benchmark = stratified_sample(
        rows=rows,
        target_by_type=DEFAULT_TARGET,
        required_inputs=DEFAULT_REQUIRED_INPUTS,
        seed=args.seed,
    )
    write_jsonl(Path(args.output), benchmark)

    stats: Dict[str, int] = {t: 0 for t in TYPE_ORDER}
    for row in benchmark:
        t = parse_type(str(row.get("output", "")))
        if t in stats:
            stats[t] += 1
    covered_required = sum(
        1
        for r in benchmark
        if str(r.get("input", "")).strip() in DEFAULT_REQUIRED_INPUTS
    )

    print("===== Fixed Benchmark Built =====")
    print(f"input_rows: {len(rows)}")
    print(f"output_rows: {len(benchmark)}")
    print(f"type_distribution: {json.dumps(stats, ensure_ascii=False)}")
    print(f"required_covered: {covered_required}/{len(DEFAULT_REQUIRED_INPUTS)}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
