#!/usr/bin/env python3
"""Prepare DPO v7.4 by adding tense-precision hardfix pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


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


def format_output(error_type: str, correction: str, explanation: str) -> str:
    return f"**错误类型**: {error_type}\n**改正**: {correction}\n**解释**: {explanation}"


def make_pair(
    sentence: str,
    chosen_type: str,
    chosen_corr: str,
    chosen_exp: str,
    rejected_type: str,
    rejected_corr: str,
    rejected_exp: str,
) -> Dict:
    return {
        "prompt": f"纠正以下句子的语法错误\n{sentence}",
        "chosen": format_output(chosen_type, chosen_corr, chosen_exp),
        "rejected": format_output(rejected_type, rejected_corr, rejected_exp),
        "pair_source": "hardfix_v7_4_tense_precision",
    }


def _key(row: Dict) -> Tuple[str, str, str]:
    return str(row.get("prompt", "")), str(row.get("chosen", "")), str(row.get("rejected", ""))


def dedupe_new_rows(new_rows: List[Dict], base_rows: List[Dict]) -> List[Dict]:
    seen: Set[Tuple[str, str, str]] = {_key(r) for r in base_rows}
    out: List[Dict] = []
    for r in new_rows:
        k = _key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def build_tense_precision_pairs() -> List[Dict]:
    pairs: List[Dict] = []

    # Pattern A: reported speech future-in-past should often be simple "was/were + V-ing" in these curated items.
    pattern_a = [
        (
            "She told me she will be going to the store.",
            "She told me she was going to the store.",
            "She told me she would be going to the store.",
        ),
        (
            "He said he will be attending the workshop the next day.",
            "He said he was attending the workshop the next day.",
            "He said he would be attending the workshop the next day.",
        ),
        (
            "They told us they will be joining the project later.",
            "They told us they were joining the project later.",
            "They told us they would be joining the project later.",
        ),
    ]
    for src, good, bad in pattern_a:
        pairs.append(
            make_pair(
                sentence=src,
                chosen_type="时态一致",
                chosen_corr=good,
                chosen_exp="在该转述语境中，后文更自然地回溯为过去进行时表达。",
                rejected_type="时态一致",
                rejected_corr=bad,
                rejected_exp="该改法虽可行但不符合本数据集目标标注的首选表达。",
            )
        )

    # Pattern B: second conditional prefers "would + V" rather than "would have + Vpp".
    pattern_b = [
        (
            "If I met him earlier, I give him a gift.",
            "If I met him earlier, I would give him a gift.",
            "If I met him earlier, I would have given him a gift.",
        ),
        (
            "If she came tomorrow, I prepare everything in advance.",
            "If she came tomorrow, I would prepare everything in advance.",
            "If she came tomorrow, I would have prepared everything in advance.",
        ),
        (
            "If they had more time now, they finish the report today.",
            "If they had more time now, they would finish the report today.",
            "If they had more time now, they would have finished the report today.",
        ),
    ]
    for src, good, bad in pattern_b:
        pairs.append(
            make_pair(
                sentence=src,
                chosen_type="时态一致",
                chosen_corr=good,
                chosen_exp="该句表达与现在事实相反的假设，主句应使用 would + 动词原形。",
                rejected_type="时态一致",
                rejected_corr=bad,
                rejected_exp="would have done 更偏向过去未实现条件，不符合该句语义。",
            )
        )

    # Pattern C: since + time point often aligns with present perfect in reported relation here.
    pattern_c = [
        (
            "He said he had been feeling sick since morning.",
            "He said he has been feeling sick since morning.",
            "He said he had been feeling sick since morning.",
        ),
        (
            "She said she had worked here since 2020.",
            "She said she has worked here since 2020.",
            "She said she had worked here since 2020.",
        ),
        (
            "They said they had lived in the city since childhood.",
            "They said they have lived in the city since childhood.",
            "They said they had lived in the city since childhood.",
        ),
    ]
    for src, good, bad in pattern_c:
        pairs.append(
            make_pair(
                sentence=src,
                chosen_type="时态一致",
                chosen_corr=good,
                chosen_exp="该句强调从过去持续到当前的状态，标注目标采用现在完成表达。",
                rejected_type="时态一致",
                rejected_corr=bad,
                rejected_exp="该改法保留了历史误判时态，不符合当前标注目标。",
            )
        )

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DPO v7.4 tense-precision hardfix data.")
    parser.add_argument("--base-train", default="data/processed_v6/dpo_train_v7_3.jsonl")
    parser.add_argument("--base-val", default="data/processed_v6/dpo_val_v7_3.jsonl")
    parser.add_argument("--hardfix-train-out", default="data/processed_v6/dpo_hardfix_train_v7_4.jsonl")
    parser.add_argument("--hardfix-val-out", default="data/processed_v6/dpo_hardfix_val_v7_4.jsonl")
    parser.add_argument("--merged-train-out", default="data/processed_v6/dpo_train_v7_4.jsonl")
    parser.add_argument("--merged-val-out", default="data/processed_v6/dpo_val_v7_4.jsonl")
    parser.add_argument("--max-val-add", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base_train = read_jsonl(Path(args.base_train))
    base_val = read_jsonl(Path(args.base_val))

    hardfix_all = build_tense_precision_pairs()
    hardfix_train = dedupe_new_rows(hardfix_all, base_train)
    hardfix_val_pool = dedupe_new_rows(hardfix_all, base_val)
    rng.shuffle(hardfix_val_pool)
    hardfix_val = hardfix_val_pool[: args.max_val_add]

    merged_train = base_train + hardfix_train
    merged_val = base_val + hardfix_val
    rng.shuffle(merged_train)
    rng.shuffle(merged_val)

    write_jsonl(Path(args.hardfix_train_out), hardfix_train)
    write_jsonl(Path(args.hardfix_val_out), hardfix_val)
    write_jsonl(Path(args.merged_train_out), merged_train)
    write_jsonl(Path(args.merged_val_out), merged_val)

    print("===== DPO v7.4 Tense Precision Prepared =====")
    print(f"base_train: {len(base_train)} -> merged_train: {len(merged_train)}")
    print(f"base_val: {len(base_val)} -> merged_val: {len(merged_val)}")
    print(f"hardfix_train_added: {len(hardfix_train)}")
    print(f"hardfix_val_added: {len(hardfix_val)}")
    print(f"wrote: {args.hardfix_train_out}")
    print(f"wrote: {args.hardfix_val_out}")
    print(f"wrote: {args.merged_train_out}")
    print(f"wrote: {args.merged_val_out}")


if __name__ == "__main__":
    main()

