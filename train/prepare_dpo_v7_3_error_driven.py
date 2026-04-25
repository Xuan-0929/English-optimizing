#!/usr/bin/env python3
"""Build DPO v7.3 error-driven hardfix pairs from eval outputs.

Focus:
- Keep only non-relaxed correction errors and type errors.
- Skip likely OCR/noise-label anomalies that may inject wrong supervision.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


WS_RE = re.compile(r"\s+")
PUNCT_WS_RE = re.compile(r"\s+([,.;:!?])")
TYPE_RE = re.compile(r"\*\*错误类型\*\*:\s*(.+?)\n")
CORR_RE = re.compile(r"\*\*改正\*\*:\s*(.+?)\n", re.S)

RELAXED_TOKEN_MAP = [
    (re.compile(r"\bit's\b"), "it is"),
    (re.compile(r"\bwho\b"), "<rel_pron>"),
    (re.compile(r"\bthat\b"), "<rel_pron>"),
    (re.compile(r"\btopic\b"), "subject"),
]


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower().replace("“", '"').replace("”", '"').replace("’", "'")
    s = WS_RE.sub(" ", s)
    return s.strip(" .;:")


def normalize_text_relaxed(s: str) -> str:
    s = normalize_text(s)
    for pat, rep in RELAXED_TOKEN_MAP:
        s = pat.sub(rep, s)
    s = PUNCT_WS_RE.sub(r"\1", s)
    s = WS_RE.sub(" ", s)
    return s.strip(" .;:")


def format_output(error_type: str, correction: str, explanation: str) -> str:
    return f"**错误类型**: {error_type}\n**改正**: {correction}\n**解释**: {explanation}"


def parse_type_and_corr(text: str) -> Tuple[str, str]:
    mt = TYPE_RE.search(text or "")
    mc = CORR_RE.search(text or "")
    return (mt.group(1).strip() if mt else "", mc.group(1).strip() if mc else "")


def likely_ocr_noise(sentence: str) -> bool:
    s = sentence or ""
    # Typical OCR confusion: ll/II mixed in alphabetic token (e.g. chlorophyII).
    # Match II at middle or token end.
    return bool(re.search(r"[A-Za-z](II)([A-Za-z]|\b)", s))


def build_hardfix_rows(eval_items: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for item in eval_items:
        sentence = str(item.get("input", "")).strip()
        gold_corr = str(item.get("gold_correction", "")).strip()
        gold_type = str(item.get("gold_type", "")).strip()
        lora_raw = str(item.get("lora_output", "")).strip()
        lora_corr = str(item.get("lora_correction", "")).strip()
        lora_type = str(item.get("lora_type", "")).strip()
        if not sentence or not gold_corr or not gold_type:
            continue
        # Skip likely OCR-noise type labels to avoid injecting questionable supervision.
        if likely_ocr_noise(sentence):
            continue

        corr_err = normalize_text(lora_corr) != normalize_text(gold_corr)
        corr_err_relaxed = normalize_text_relaxed(lora_corr) != normalize_text_relaxed(gold_corr)
        type_err = normalize_text(lora_type) != normalize_text(gold_type)
        # Only keep true (non-relaxed) correction errors OR type errors.
        if not (type_err or corr_err_relaxed):
            continue

        rejected_type, rejected_corr = parse_type_and_corr(lora_raw)
        if not rejected_type:
            rejected_type = lora_type or "时态一致"
        if not rejected_corr:
            rejected_corr = lora_corr or sentence

        chosen = format_output(
            gold_type,
            gold_corr,
            "基于误差驱动修复：选择与标注一致的错误类型与改正句。",
        )
        rejected = format_output(
            rejected_type,
            rejected_corr,
            "该候选来自历史误判，保留为对比负样本。",
        )
        if chosen == rejected:
            continue
        rows.append(
            {
                "prompt": f"纠正以下句子的语法错误\n{sentence}",
                "chosen": chosen,
                "rejected": rejected,
                "pair_source": "hardfix_v7_3_error_driven",
                "error_id": item.get("id"),
            }
        )
    # de-dup by (prompt,chosen,rejected)
    seen = set()
    out = []
    for r in rows:
        k = (r["prompt"], r["chosen"], r["rejected"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def merge(base_rows: List[Dict], add_rows: List[Dict]) -> List[Dict]:
    seen = {(str(r.get("prompt", "")), str(r.get("chosen", "")), str(r.get("rejected", ""))) for r in base_rows}
    merged = list(base_rows)
    for r in add_rows:
        k = (str(r.get("prompt", "")), str(r.get("chosen", "")), str(r.get("rejected", "")))
        if k in seen:
            continue
        seen.add(k)
        merged.append(r)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare DPO v7.3 error-driven hardfix dataset.")
    ap.add_argument("--base-train", default="data/processed_v6/dpo_train_v7_2.jsonl")
    ap.add_argument("--base-val", default="data/processed_v6/dpo_val_v7_2.jsonl")
    ap.add_argument("--eval-ood", default="result/dpo_v7_2_constraint/ood_eval_compare.json")
    ap.add_argument("--eval-noise", default="result/dpo_v7_2_constraint/noise_robust_eval_compare.json")
    ap.add_argument("--hardfix-train-out", default="data/processed_v6/dpo_hardfix_train_v7_3.jsonl")
    ap.add_argument("--hardfix-val-out", default="data/processed_v6/dpo_hardfix_val_v7_3.jsonl")
    ap.add_argument("--merged-train-out", default="data/processed_v6/dpo_train_v7_3.jsonl")
    ap.add_argument("--merged-val-out", default="data/processed_v6/dpo_val_v7_3.jsonl")
    args = ap.parse_args()

    base_train = read_jsonl(Path(args.base_train))
    base_val = read_jsonl(Path(args.base_val))
    ood_items = read_json(Path(args.eval_ood)).get("items", [])
    noise_items = read_json(Path(args.eval_noise)).get("items", [])

    all_hardfix = build_hardfix_rows(ood_items + noise_items)
    # Small-sample regime: put all hardfix into train, and a small representative subset into val.
    hardfix_train = all_hardfix
    hardfix_val = all_hardfix[: max(1, min(4, len(all_hardfix)))]

    merged_train = merge(base_train, hardfix_train)
    merged_val = merge(base_val, hardfix_val)

    write_jsonl(Path(args.hardfix_train_out), hardfix_train)
    write_jsonl(Path(args.hardfix_val_out), hardfix_val)
    write_jsonl(Path(args.merged_train_out), merged_train)
    write_jsonl(Path(args.merged_val_out), merged_val)

    print("===== DPO v7.3 Error-driven Hardfix Prepared =====")
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
