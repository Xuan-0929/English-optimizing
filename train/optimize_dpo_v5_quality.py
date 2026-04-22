#!/usr/bin/env python3
"""Quality optimization for DPO v5 dataset.

Actions:
1) Remove train rows that leak sft validation inputs.
2) Remove clearly noisy subject-verb agreement rules (`or/nor + singular -> are`).
3) De-duplicate by prompt, keeping the highest-quality pair.
4) Reduce low-signal type-only pairs (same correction, only type differs), except hardfix.
5) Apply the same structural cleanup to val (without leakage filtering).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TYPE_RE = re.compile(r"\*\*错误类型\*\*:\s*(.+?)\n")
CORR_RE = re.compile(r"\*\*改正\*\*:\s*(.+?)\n", re.S)


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_type_and_corr(text: str) -> Optional[Tuple[str, str]]:
    m_type = TYPE_RE.search(text)
    m_corr = CORR_RE.search(text)
    if not (m_type and m_corr):
        return None
    return m_type.group(1).strip(), m_corr.group(1).strip()


def prompt_input(prompt: str) -> str:
    return prompt.split("\n", 1)[1].strip() if "\n" in prompt else prompt.strip()


def is_noisy_sva_prompt(src: str, chosen_corr: str) -> bool:
    s = src.strip()
    c = chosen_corr.strip()

    # `Neither A nor B is ...` is usually acceptable when near subject is singular.
    if s.startswith("Neither the ") and " nor the " in s and " is " in s and " are " in c:
        return True

    # `A or B is ...` with singular alternatives should not be blindly changed to `are`.
    if " or the " in s and " is " in s and " are " in c:
        return True

    return False


def pair_quality_score(row: Dict) -> int:
    chosen = str(row.get("chosen", ""))
    rejected = str(row.get("rejected", ""))
    source = str(row.get("pair_source", "base"))
    parsed_c = parse_type_and_corr(chosen)
    parsed_r = parse_type_and_corr(rejected)
    if not (parsed_c and parsed_r):
        return -10

    t_c, corr_c = parsed_c
    t_r, corr_r = parsed_r
    tdiff = t_c != t_r
    cdiff = corr_c != corr_r

    score = 0
    if cdiff:
        score += 5
    if tdiff:
        score += 2

    if source in {"base", "hardfix_v5", "targeted_v4_natural"}:
        score += 2
    if source in {"type_consistency", "type_focus_v3", "targeted_v4_synth"}:
        score -= 2

    if "句子已正确，无需修改" in rejected:
        score -= 1

    if is_noisy_sva_prompt(prompt_input(str(row.get("prompt", ""))), corr_c):
        score -= 8

    return score


def filter_rows(
    rows: List[Dict],
    leakage_inputs: Optional[set[str]],
    keep_hardfix_type_only: bool,
    stats: Counter,
) -> List[Dict]:
    out: List[Dict] = []
    for row in rows:
        prompt = str(row.get("prompt", ""))
        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))
        source = str(row.get("pair_source", "base"))
        inp = prompt_input(prompt)

        if leakage_inputs is not None and inp in leakage_inputs:
            stats["drop_leakage"] += 1
            continue

        parsed_c = parse_type_and_corr(chosen)
        parsed_r = parse_type_and_corr(rejected)
        if not (parsed_c and parsed_r):
            stats["drop_parse_fail"] += 1
            continue

        t_c, corr_c = parsed_c
        t_r, corr_r = parsed_r
        tdiff = t_c != t_r
        cdiff = corr_c != corr_r

        if is_noisy_sva_prompt(inp, corr_c):
            stats["drop_noisy_sva"] += 1
            continue

        # Low-signal pair: same correction, only type differs.
        if tdiff and not cdiff:
            if keep_hardfix_type_only and source == "hardfix_v5":
                pass
            else:
                stats["drop_type_only"] += 1
                continue

        out.append(row)
    return out


def dedupe_by_prompt(rows: List[Dict], stats: Counter) -> List[Dict]:
    best: Dict[str, Tuple[int, Dict]] = {}
    for row in rows:
        p = str(row.get("prompt", ""))
        score = pair_quality_score(row)
        old = best.get(p)
        if old is None or score > old[0]:
            if old is not None:
                stats["drop_dup_prompt"] += 1
            best[p] = (score, row)
        else:
            stats["drop_dup_prompt"] += 1
    cleaned = [v[1] for v in best.values()]
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize DPO v5 dataset quality.")
    parser.add_argument("--train-in", default="data/processed_v6/dpo_train_v5.jsonl")
    parser.add_argument("--val-in", default="data/processed_v6/dpo_val_v5.jsonl")
    parser.add_argument("--sft-val", default="data/processed_v6/sft_train_v6_val.jsonl")
    parser.add_argument("--train-out", default="data/processed_v6/dpo_train_v6_qc.jsonl")
    parser.add_argument("--val-out", default="data/processed_v6/dpo_val_v6_qc.jsonl")
    parser.add_argument("--report-out", default="data/processed_v6/dpo_v6_qc_report.json")
    args = parser.parse_args()

    train_rows = read_jsonl(Path(args.train_in))
    val_rows = read_jsonl(Path(args.val_in))
    sft_val_rows = read_jsonl(Path(args.sft_val))
    leak_inputs = {str(r.get("input", "")).strip() for r in sft_val_rows if str(r.get("input", "")).strip()}

    train_stats: Counter = Counter()
    val_stats: Counter = Counter()

    train_filtered = filter_rows(
        train_rows,
        leakage_inputs=leak_inputs,
        keep_hardfix_type_only=True,
        stats=train_stats,
    )
    val_filtered = filter_rows(
        val_rows,
        leakage_inputs=None,
        keep_hardfix_type_only=True,
        stats=val_stats,
    )

    train_clean = dedupe_by_prompt(train_filtered, train_stats)
    val_clean = dedupe_by_prompt(val_filtered, val_stats)

    write_jsonl(Path(args.train_out), train_clean)
    write_jsonl(Path(args.val_out), val_clean)

    report = {
        "train": {
            "input_rows": len(train_rows),
            "output_rows": len(train_clean),
            "stats": dict(train_stats),
        },
        "val": {
            "input_rows": len(val_rows),
            "output_rows": len(val_clean),
            "stats": dict(val_stats),
        },
        "output_files": {
            "train": args.train_out,
            "val": args.val_out,
            "report": args.report_out,
        },
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== DPO v6 QC Done =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
