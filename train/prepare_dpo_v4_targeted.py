#!/usr/bin/env python3
"""Prepare DPO v4 data with targeted fixes and capped type-consistency ratio.

Goals:
1. Add targeted pairs for known misses:
   - `married with` -> `married`
   - `researches has` -> `research has`
2. Keep type-only pairs (`type_consistency` / `type_focus_v3`) <= target ratio.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


TYPE_RE = re.compile(r"\*\*错误类型\*\*:\s*(.+?)\n")
CORR_RE = re.compile(r"\*\*改正\*\*:\s*(.+?)\n", re.S)
EXP_RE = re.compile(r"\*\*解释\*\*:\s*(.+)$", re.S)

DEFAULT_INSTRUCTION = "纠正以下句子的语法错误"
CONSISTENCY_SOURCES = {"type_consistency", "type_focus_v3"}


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


def make_type_pair(
    prompt: str,
    error_type: str,
    correction: str,
    explanation: str,
    wrong_type: str,
    pair_source: str,
) -> Optional[Dict]:
    chosen = format_output(error_type, correction, explanation)
    rejected = format_output(wrong_type, correction, explanation)
    if chosen == rejected:
        return None
    return {
        "prompt": prompt.strip(),
        "chosen": chosen,
        "rejected": rejected,
        "pair_source": pair_source,
    }


def _is_target_input(text: str) -> bool:
    t = text.lower()
    return "married with" in t or "researches has" in t


def _key(row: Dict) -> Tuple[str, str, str]:
    return str(row.get("prompt", "")), str(row.get("chosen", "")), str(row.get("rejected", ""))


def build_pairs_from_sft_rows(
    sft_rows: List[Dict],
    wrong_type: str,
    pair_source: str,
) -> List[Dict]:
    pairs: List[Dict] = []
    for row in sft_rows:
        instruction = str(row.get("instruction", "")).strip()
        user_input = str(row.get("input", "")).strip()
        output = str(row.get("output", "")).strip()
        if not instruction or not user_input or not _is_target_input(user_input):
            continue
        parsed = parse_output(output)
        if parsed is None:
            continue
        error_type, correction, explanation = parsed
        prompt = f"{instruction}\n{user_input}".strip()
        pair = make_type_pair(
            prompt=prompt,
            error_type=error_type,
            correction=correction,
            explanation=explanation,
            wrong_type=wrong_type,
            pair_source=pair_source,
        )
        if pair is not None:
            pairs.append(pair)
    return pairs


def build_synthetic_married_with_pairs(pair_source: str) -> List[Dict]:
    wrong_sentences = [
        "She married with her college classmate in 2014.",
        "He married with his longtime partner last spring.",
        "They married with each other after graduation.",
        "My cousin married with a doctor from Shanghai.",
        "Our neighbor married with her best friend in 2020.",
        "The actor married with a singer in a private ceremony.",
        "Linda married with her fiance in June.",
        "He married with someone he met at university.",
        "Sara married with her boyfriend after five years.",
        "The captain married with his childhood friend last year.",
        "She married with a French engineer in Paris.",
        "Tom married with his partner during the holiday.",
        "They finally married with each other in autumn.",
        "My aunt married with an artist from Milan.",
        "He married with his girlfriend right after college.",
        "The teacher married with a journalist in 2016.",
        "She married with him despite her parents' concerns.",
        "Our manager married with her fiance last weekend.",
    ]
    explanation = "marry 为及物动词，后面不加 with。"
    out: List[Dict] = []
    for sent in wrong_sentences:
        correction = sent.replace("married with ", "married ", 1)
        prompt = f"{DEFAULT_INSTRUCTION}\n{sent}"
        pair = make_type_pair(
            prompt=prompt,
            error_type="介词",
            correction=correction,
            explanation=explanation,
            wrong_type="时态一致",
            pair_source=pair_source,
        )
        if pair is not None:
            out.append(pair)
    return out


def build_synthetic_researches_has_pairs(pair_source: str) -> List[Dict]:
    wrong_sentences = [
        "Clinical researches has changed hospital protocols.",
        "The new researches has improved diagnosis accuracy.",
        "Applied researches has strong commercial value.",
        "Current researches has influenced policy decisions.",
        "Behavioral researches has revealed useful patterns.",
        "Linguistic researches has clarified this phenomenon.",
        "Educational researches has guided curriculum design.",
        "Neuroscience researches has expanded our understanding.",
        "Field researches has provided reliable evidence.",
        "Market researches has reshaped our strategy.",
        "Historical researches has corrected earlier assumptions.",
        "Pilot researches has shown promising outcomes.",
        "Long-term researches has reduced uncertainty.",
        "Comparative researches has highlighted key differences.",
        "Independent researches has supported the same conclusion.",
        "Recent clinical researches has improved treatment plans.",
        "Advanced AI researches has accelerated product iteration.",
        "The follow-up researches has confirmed the hypothesis.",
    ]
    explanation = "research 在此语境下通常作不可数名词，谓语动词应使用单数形式。"
    out: List[Dict] = []
    for sent in wrong_sentences:
        correction = sent.replace("researches", "research", 1)
        prompt = f"{DEFAULT_INSTRUCTION}\n{sent}"
        pair = make_type_pair(
            prompt=prompt,
            error_type="主谓一致",
            correction=correction,
            explanation=explanation,
            wrong_type="时态一致",
            pair_source=pair_source,
        )
        if pair is not None:
            out.append(pair)
    return out


def dedupe_new_pairs(new_pairs: List[Dict], existing: List[Dict]) -> List[Dict]:
    seen: Set[Tuple[str, str, str]] = {_key(r) for r in existing}
    deduped: List[Dict] = []
    for row in new_pairs:
        k = _key(row)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(row)
    return deduped


def cap_consistency_rows(
    train_rows: List[Dict],
    rng: random.Random,
    max_ratio: float,
) -> Tuple[List[Dict], int, int]:
    non_cons = [r for r in train_rows if str(r.get("pair_source", "")) not in CONSISTENCY_SOURCES]
    cons = [r for r in train_rows if str(r.get("pair_source", "")) in CONSISTENCY_SOURCES]
    before = len(cons)
    if not train_rows:
        return train_rows, before, before
    # Need keep_cons / (non_cons + keep_cons) <= max_ratio.
    # Rearranged: keep_cons <= non_cons * max_ratio / (1 - max_ratio).
    if max_ratio >= 1.0:
        keep_cons = len(cons)
    else:
        keep_cons = int((len(non_cons) * max_ratio) / (1.0 - max_ratio))
    keep_cons = min(keep_cons, len(cons))
    rng.shuffle(cons)
    kept = cons[:keep_cons]
    after_rows = non_cons + kept
    rng.shuffle(after_rows)
    return after_rows, before, len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare targeted DPO v4 dataset.")
    parser.add_argument("--sft-train", default="data/processed_v6/sft_train_v6_train.jsonl")
    parser.add_argument("--sft-val", default="data/processed_v6/sft_train_v6_val.jsonl")
    parser.add_argument("--dpo-train-base", default="data/processed_v6/dpo_train_v3.jsonl")
    parser.add_argument("--dpo-val-base", default="data/processed_v6/dpo_val_v3.jsonl")
    parser.add_argument("--target-train-out", default="data/processed_v6/dpo_target_train_v4.jsonl")
    parser.add_argument("--target-val-out", default="data/processed_v6/dpo_target_val_v4.jsonl")
    parser.add_argument("--merged-train-out", default="data/processed_v6/dpo_train_v4.jsonl")
    parser.add_argument("--merged-val-out", default="data/processed_v6/dpo_val_v4.jsonl")
    parser.add_argument("--max-target-train", type=int, default=32)
    parser.add_argument("--max-target-val", type=int, default=8)
    parser.add_argument("--max-consistency-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0.0 <= args.max_consistency_ratio <= 1.0):
        raise ValueError("--max-consistency-ratio must be within [0, 1].")

    rng = random.Random(args.seed)
    sft_train_rows = read_jsonl(Path(args.sft_train))
    sft_val_rows = read_jsonl(Path(args.sft_val))
    base_train_rows = read_jsonl(Path(args.dpo_train_base))
    base_val_rows = read_jsonl(Path(args.dpo_val_base))

    natural_train = build_pairs_from_sft_rows(
        sft_rows=sft_train_rows,
        wrong_type="时态一致",
        pair_source="targeted_v4_natural",
    )
    natural_val = build_pairs_from_sft_rows(
        sft_rows=sft_val_rows,
        wrong_type="时态一致",
        pair_source="targeted_v4_natural",
    )
    synthetic = build_synthetic_married_with_pairs("targeted_v4_synth") + build_synthetic_researches_has_pairs("targeted_v4_synth")
    rng.shuffle(synthetic)

    natural_train = dedupe_new_pairs(natural_train, base_train_rows)
    train_pool = dedupe_new_pairs(natural_train + synthetic, base_train_rows)
    rng.shuffle(train_pool)
    target_train_rows = train_pool[: args.max_target_train]

    natural_val = dedupe_new_pairs(natural_val, base_val_rows)
    val_pool = dedupe_new_pairs(natural_val + synthetic, base_val_rows)
    rng.shuffle(val_pool)
    target_val_rows = val_pool[: args.max_target_val]

    merged_train = base_train_rows + target_train_rows
    merged_val = base_val_rows + target_val_rows
    merged_train, cons_before, cons_after = cap_consistency_rows(
        train_rows=merged_train,
        rng=rng,
        max_ratio=args.max_consistency_ratio,
    )

    write_jsonl(Path(args.target_train_out), target_train_rows)
    write_jsonl(Path(args.target_val_out), target_val_rows)
    write_jsonl(Path(args.merged_train_out), merged_train)
    write_jsonl(Path(args.merged_val_out), merged_val)

    print("===== DPO Targeted v4 Prepared =====")
    print(f"base_train: {len(base_train_rows)} -> merged_train: {len(merged_train)}")
    print(f"base_val: {len(base_val_rows)} -> merged_val: {len(merged_val)}")
    print(f"target_train_added: {len(target_train_rows)}")
    print(f"target_val_added: {len(target_val_rows)}")
    print(f"consistency_rows_before_cap: {cons_before}")
    print(f"consistency_rows_after_cap: {cons_after}")
    if merged_train:
        print(f"consistency_ratio_after_cap: {cons_after / len(merged_train):.4f}")
    print(f"wrote: {args.target_train_out}")
    print(f"wrote: {args.target_val_out}")
    print(f"wrote: {args.merged_train_out}")
    print(f"wrote: {args.merged_val_out}")


if __name__ == "__main__":
    main()
