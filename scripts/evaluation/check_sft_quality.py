"""一次性检查 SFT 训练集质量（不依赖 analyze_dataset 的旧默认路径）。"""
import json
import re
import os
import sys
from collections import Counter, defaultdict


def analyze(file_path: str) -> dict:
    total = 0
    error_type_counter: Counter = Counter()
    duplicate_inputs: defaultdict = defaultdict(int)
    same_in_out = 0
    missing_fields = 0
    bad_format = 0
    empty_io = 0
    missing_output_pattern = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                bad_format += 1
                continue
            total += 1
            inp = entry.get("input", "")
            out = entry.get("output", "")
            if not inp or not out:
                empty_io += 1
            duplicate_inputs[inp] += 1
            if "instruction" not in entry or "input" not in entry or "output" not in entry:
                missing_fields += 1
            has_type = "**错误类型**" in out
            has_fix = "**改正**" in out
            has_exp = "**解释**" in out
            if not (has_type and has_fix and has_exp):
                missing_output_pattern += 1
            m_type = re.search(r"\*\*错误类型\*\*:\s*(.*?)\n", out)
            m_fix = re.search(r"\*\*改正\*\*:\s*(.*?)\n", out)
            if m_type:
                error_type_counter[m_type.group(1).strip()] += 1
            else:
                error_type_counter["(无错误类型行)"] += 1
            corrected = m_fix.group(1).strip() if m_fix else ""
            if inp.strip().lower() == corrected.strip().lower() and inp:
                same_in_out += 1

    dup_inputs = sum(1 for c in duplicate_inputs.values() if c > 1)
    multi_dup = sum(c - 1 for c in duplicate_inputs.values() if c > 1)
    return {
        "file": file_path,
        "total": total,
        "bad_json": bad_format,
        "missing_fields": missing_fields,
        "empty_io": empty_io,
        "dup_unique_inputs": dup_inputs,
        "extra_dup_rows": multi_dup,
        "same_error_as_fix": same_in_out,
        "error_types": error_type_counter,
        "missing_triple_star": missing_output_pattern,
    }


def main():
    paths = sys.argv[1:] or ["data/sft_train.jsonl", "data/sft_train_filtered.jsonl"]
    for p in paths:
        if not os.path.exists(p):
            print(f"跳过（不存在）: {p}")
            continue
        r = analyze(p)
        print(f"=== {p} ===")
        print(
            f"总样本: {r['total']} | JSON坏行: {r['bad_json']} | 缺字段: {r['missing_fields']} | input/output空: {r['empty_io']}"
        )
        print(f"缺 **错误类型/改正/解释** 子串任一: {r['missing_triple_star']}")
        print(f"重复 input 种类数: {r['dup_unique_inputs']} | 因重复多出的行数: {r['extra_dup_rows']}")
        print(f"错句与改正相同(疑似无效样本): {r['same_error_as_fix']}")
        print("错误类型 Top 15:")
        for k, v in r["error_types"].most_common(15):
            pct = 100 * v / r["total"] if r["total"] else 0
            print(f"  {k}: {v} ({pct:.1f}%)")
        print()


if __name__ == "__main__":
    main()
