#!/usr/bin/env python3
"""Build extended evaluation suites beyond the fixed benchmark.

Outputs:
- challenge_type_v1: type-confusion-focused set
- ood_general_v1: broader out-of-distribution style set
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set


TYPE_RE = re.compile(r"\*\*错误类型\*\*:\s*(.+?)\n")
TYPE_ORDER = ["介词", "主谓一致", "时态一致", "定语从句"]

CHALLENGE_PATTERNS = [
    r"\bmarried with\b",
    r"\bfamiliar of\b",
    r"\bagree with .+ at\b",
    r"\bconcur with .+ at\b",
    r"\bon next\b",
    r"\bused to\b",
    r"\bwhich its\b",
    r"\bthat he\b",
    r"\bneither\b.+\bnor\b",
    r"\bor the\b.+\bis\b",
]


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
    return m.group(1).strip() if m else None


def input_key(row: Dict) -> str:
    return str(row.get("input", "")).strip()


def select_challenge_rows(
    rows: List[Dict],
    exclude_inputs: Set[str],
    target_size: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    patt = [re.compile(p, re.I) for p in CHALLENGE_PATTERNS]
    matched = []
    for r in rows:
        inp = input_key(r)
        if not inp or inp in exclude_inputs:
            continue
        if any(p.search(inp) for p in patt):
            matched.append(r)
    rng.shuffle(matched)
    out: List[Dict] = []
    seen: Set[str] = set()
    for r in matched:
        k = input_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
        if len(out) >= target_size:
            break
    return out


def stratified_sample(
    rows: List[Dict],
    exclude_inputs: Set[str],
    target_size: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    per_type = max(1, target_size // len(TYPE_ORDER))
    buckets: Dict[str, List[Dict]] = {t: [] for t in TYPE_ORDER}
    for r in rows:
        inp = input_key(r)
        if not inp or inp in exclude_inputs:
            continue
        t = parse_type(str(r.get("output", "")))
        if t in buckets:
            buckets[t].append(r)
    for t in TYPE_ORDER:
        rng.shuffle(buckets[t])

    out: List[Dict] = []
    seen: Set[str] = set()
    for t in TYPE_ORDER:
        for r in buckets[t]:
            k = input_key(r)
            if k in seen:
                continue
            out.append(r)
            seen.add(k)
            if sum(1 for x in out if parse_type(str(x.get("output", ""))) == t) >= per_type:
                break

    pool = [r for t in TYPE_ORDER for r in buckets[t]]
    rng.shuffle(pool)
    for r in pool:
        if len(out) >= target_size:
            break
        k = input_key(r)
        if k in seen:
            continue
        out.append(r)
        seen.add(k)
    return out


def type_stats(rows: List[Dict]) -> Dict[str, int]:
    stats = {t: 0 for t in TYPE_ORDER}
    for r in rows:
        t = parse_type(str(r.get("output", "")))
        if t in stats:
            stats[t] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Build challenge/ood evaluation suites.")
    ap.add_argument("--input-clean", default="data/processed_v6/sft_train_clean_v6.jsonl")
    ap.add_argument("--benchmark", default="data/processed_v6/sft_eval_benchmark_v1.jsonl")
    ap.add_argument("--challenge-out", default="data/processed_v6/sft_eval_challenge_type_v1.jsonl")
    ap.add_argument("--ood-out", default="data/processed_v6/sft_eval_ood_general_v1.jsonl")
    ap.add_argument("--report-out", default="data/processed_v6/sft_eval_suites_report_v1.json")
    ap.add_argument("--challenge-size", type=int, default=100)
    ap.add_argument("--ood-size", type=int, default=220)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clean_rows = read_jsonl(Path(args.input_clean))
    benchmark_rows = read_jsonl(Path(args.benchmark))
    benchmark_inputs = {input_key(r) for r in benchmark_rows if input_key(r)}

    challenge_rows = select_challenge_rows(
        rows=clean_rows,
        exclude_inputs=benchmark_inputs,
        target_size=args.challenge_size,
        seed=args.seed,
    )
    challenge_inputs = {input_key(r) for r in challenge_rows}
    ood_rows = stratified_sample(
        rows=clean_rows,
        exclude_inputs=benchmark_inputs.union(challenge_inputs),
        target_size=args.ood_size,
        seed=args.seed + 1,
    )

    write_jsonl(Path(args.challenge_out), challenge_rows)
    write_jsonl(Path(args.ood_out), ood_rows)

    report = {
        "input_clean_rows": len(clean_rows),
        "benchmark_rows": len(benchmark_rows),
        "challenge_rows": len(challenge_rows),
        "ood_rows": len(ood_rows),
        "challenge_type_distribution": type_stats(challenge_rows),
        "ood_type_distribution": type_stats(ood_rows),
        "output_files": {
            "challenge": args.challenge_out,
            "ood": args.ood_out,
            "report": args.report_out,
        },
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== Eval Suites Built =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
