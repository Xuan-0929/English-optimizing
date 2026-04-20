#!/usr/bin/env python3
"""Prepare DPO v5 data with hard-fix pairs for known stubborn failures.

Focus targets:
1) Type mistake on `married with` prompts (predicting 时态一致 instead of 介词)
2) Correction mistake on `researches has` prompts (only changing has->have)
"""

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
    return (
        f"**错误类型**: {error_type}\n"
        f"**改正**: {correction}\n"
        f"**解释**: {explanation}"
    )


def make_pair(
    sentence: str,
    chosen_type: str,
    chosen_correction: str,
    chosen_explanation: str,
    rejected_type: str,
    rejected_correction: str,
    rejected_explanation: str,
    pair_source: str = "hardfix_v5",
) -> Dict:
    prompt = f"纠正以下句子的语法错误\n{sentence}"
    return {
        "prompt": prompt,
        "chosen": format_output(chosen_type, chosen_correction, chosen_explanation),
        "rejected": format_output(rejected_type, rejected_correction, rejected_explanation),
        "pair_source": pair_source,
    }


def _key(row: Dict) -> Tuple[str, str, str]:
    return str(row.get("prompt", "")), str(row.get("chosen", "")), str(row.get("rejected", ""))


def dedupe_new_pairs(new_rows: List[Dict], existing_rows: List[Dict]) -> List[Dict]:
    seen: Set[Tuple[str, str, str]] = {_key(r) for r in existing_rows}
    out: List[Dict] = []
    for row in new_rows:
        k = _key(row)
        if k in seen:
            continue
        seen.add(k)
        out.append(row)
    return out


def build_marry_type_pairs() -> List[Dict]:
    examples = [
        ("She married with him in 2018.", "She married him in 2018."),
        ("She married with her classmate in 2018.", "She married her classmate in 2018."),
        ("He married with his girlfriend last year.", "He married his girlfriend last year."),
        ("They married with each other after college.", "They married each other after college."),
        ("My cousin married with a doctor in 2020.", "My cousin married a doctor in 2020."),
        ("Our teacher married with a journalist in June.", "Our teacher married a journalist in June."),
        ("Linda married with her fiance in Paris.", "Linda married her fiance in Paris."),
        ("Tom married with his partner after graduation.", "Tom married his partner after graduation."),
        ("She married with a French engineer last spring.", "She married a French engineer last spring."),
        ("The actor married with a singer in a private ceremony.", "The actor married a singer in a private ceremony."),
    ]
    out: List[Dict] = []
    for wrong, fixed in examples:
        out.append(
            make_pair(
                sentence=wrong,
                chosen_type="介词",
                chosen_correction=fixed,
                chosen_explanation="marry 为及物动词，后面不加 with。",
                rejected_type="时态一致",
                rejected_correction=fixed,
                rejected_explanation="根据主句语境与时间关系调整时态。",
            )
        )
    return out


def build_research_correction_pairs() -> List[Dict]:
    # Rejected deliberately uses common wrong fix pattern: only has->have.
    examples = [
        (
            "Clinical researches has strict ethics requirements.",
            "Clinical research has strict ethics requirements.",
            "Clinical researches have strict ethics requirements.",
        ),
        (
            "The researches has changed our understanding.",
            "The research has changed our understanding.",
            "The researches have changed our understanding.",
        ),
        (
            "Recent researches has influenced policy decisions.",
            "Recent research has influenced policy decisions.",
            "Recent researches have influenced policy decisions.",
        ),
        (
            "Current researches has improved treatment outcomes.",
            "Current research has improved treatment outcomes.",
            "Current researches have improved treatment outcomes.",
        ),
        (
            "The new researches has strong practical value.",
            "The new research has strong practical value.",
            "The new researches have strong practical value.",
        ),
        (
            "Educational researches has guided curriculum design.",
            "Educational research has guided curriculum design.",
            "Educational researches have guided curriculum design.",
        ),
        (
            "Behavioral researches has revealed useful patterns.",
            "Behavioral research has revealed useful patterns.",
            "Behavioral researches have revealed useful patterns.",
        ),
        (
            "Field researches has provided key evidence.",
            "Field research has provided key evidence.",
            "Field researches have provided key evidence.",
        ),
        (
            "Historical researches has corrected earlier assumptions.",
            "Historical research has corrected earlier assumptions.",
            "Historical researches have corrected earlier assumptions.",
        ),
        (
            "Applied researches has accelerated product iteration.",
            "Applied research has accelerated product iteration.",
            "Applied researches have accelerated product iteration.",
        ),
        (
            "Neuroscience researches has expanded our understanding.",
            "Neuroscience research has expanded our understanding.",
            "Neuroscience researches have expanded our understanding.",
        ),
        (
            "Independent researches has supported this conclusion.",
            "Independent research has supported this conclusion.",
            "Independent researches have supported this conclusion.",
        ),
    ]
    out: List[Dict] = []
    for wrong, fixed, wrong_fix in examples:
        out.append(
            make_pair(
                sentence=wrong,
                chosen_type="主谓一致",
                chosen_correction=fixed,
                chosen_explanation="research 在此语境下通常作不可数名词，谓语动词应使用单数形式。",
                rejected_type="主谓一致",
                rejected_correction=wrong_fix,
                rejected_explanation="仅将谓语改为复数并不能解决主语名词误用问题。",
            )
        )
    return out


def build_hardfix_pairs() -> List[Dict]:
    return build_marry_type_pairs() + build_research_correction_pairs()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare hard-fix DPO v5 dataset.")
    parser.add_argument("--dpo-train-base", default="data/processed_v6/dpo_train_v4.jsonl")
    parser.add_argument("--dpo-val-base", default="data/processed_v6/dpo_val_v4.jsonl")
    parser.add_argument("--hardfix-train-out", default="data/processed_v6/dpo_hardfix_train_v5.jsonl")
    parser.add_argument("--hardfix-val-out", default="data/processed_v6/dpo_hardfix_val_v5.jsonl")
    parser.add_argument("--merged-train-out", default="data/processed_v6/dpo_train_v5.jsonl")
    parser.add_argument("--merged-val-out", default="data/processed_v6/dpo_val_v5.jsonl")
    parser.add_argument("--max-hardfix-train", type=int, default=20)
    parser.add_argument("--max-hardfix-val", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_base = read_jsonl(Path(args.dpo_train_base))
    val_base = read_jsonl(Path(args.dpo_val_base))

    hardfix_all = build_hardfix_pairs()
    rng.shuffle(hardfix_all)

    hardfix_train = dedupe_new_pairs(hardfix_all, train_base)[: args.max_hardfix_train]
    hardfix_val = dedupe_new_pairs(hardfix_all, val_base)[: args.max_hardfix_val]

    merged_train = train_base + hardfix_train
    merged_val = val_base + hardfix_val
    rng.shuffle(merged_train)
    rng.shuffle(merged_val)

    write_jsonl(Path(args.hardfix_train_out), hardfix_train)
    write_jsonl(Path(args.hardfix_val_out), hardfix_val)
    write_jsonl(Path(args.merged_train_out), merged_train)
    write_jsonl(Path(args.merged_val_out), merged_val)

    print("===== DPO Hardfix v5 Prepared =====")
    print(f"base_train: {len(train_base)} -> merged_train: {len(merged_train)}")
    print(f"base_val: {len(val_base)} -> merged_val: {len(merged_val)}")
    print(f"hardfix_train_added: {len(hardfix_train)}")
    print(f"hardfix_val_added: {len(hardfix_val)}")
    print(f"wrote: {args.hardfix_train_out}")
    print(f"wrote: {args.hardfix_val_out}")
    print(f"wrote: {args.merged_train_out}")
    print(f"wrote: {args.merged_val_out}")


if __name__ == "__main__":
    main()
