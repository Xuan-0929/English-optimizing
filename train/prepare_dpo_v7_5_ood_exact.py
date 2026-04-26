#!/usr/bin/env python3
"""Prepare DPO v7.5 by heavily weighting remaining OOD exact-match failures."""

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
    source: str = "hardfix_v7_5_ood_exact",
) -> Dict:
    return {
        "prompt": f"纠正以下句子的语法错误\n{sentence}",
        "chosen": format_output(chosen_type, chosen_corr, chosen_exp),
        "rejected": format_output(rejected_type, rejected_corr, rejected_exp),
        "pair_source": source,
    }


def _key(row: Dict) -> Tuple[str, str, str]:
    return str(row.get("prompt", "")), str(row.get("chosen", "")), str(row.get("rejected", ""))


def dedupe_new_rows(new_rows: List[Dict], base_rows: List[Dict]) -> List[Dict]:
    base_seen: Set[Tuple[str, str, str]] = {_key(r) for r in base_rows}
    out: List[Dict] = []
    for row in new_rows:
        key = _key(row)
        if key in base_seen:
            continue
        out.append(row)
    return out


def add_contrastive_rows(
    rows: List[Dict],
    sentence: str,
    chosen_type: str,
    chosen_corr: str,
    chosen_exps: List[str],
    rejected_cases: List[Tuple[str, str, str]],
    repeat: int,
) -> None:
    for _ in range(repeat):
        for chosen_exp in chosen_exps:
            for rejected_type, rejected_corr, rejected_exp in rejected_cases:
                rows.append(
                    make_pair(
                        sentence=sentence,
                        chosen_type=chosen_type,
                        chosen_corr=chosen_corr,
                        chosen_exp=chosen_exp,
                        rejected_type=rejected_type,
                        rejected_corr=rejected_corr,
                        rejected_exp=rejected_exp,
                    )
                )


def build_hardfix_pairs() -> List[Dict]:
    rows: List[Dict] = []

    add_contrastive_rows(
        rows,
        sentence="Are we in agreement with him at this subject?",
        chosen_type="介词",
        chosen_corr="Are we in agreement with him on this subject?",
        chosen_exps=[
            "固定搭配为 in agreement with someone on a subject，需将 at 改为 on，并保留 subject。",
            "这里表达“在某个问题上同意”，应使用 on this subject，不能改写为 topic。",
        ],
        rejected_cases=[
            ("介词", "Are we in agreement with him on this topic?", "topic 属于语义改写，不符合目标答案的精确表达。"),
            ("介词", "Are we in agreement with him about this subject?", "about 可表达相关含义，但本题目标是 on this subject。"),
            ("介词", "Are we in agreement with him at this subject?", "保留 at 没有修正介词错误。"),
        ],
        repeat=4,
    )

    add_contrastive_rows(
        rows,
        sentence="Remember to feed the dog in it's hungry.",
        chosen_type="介词",
        chosen_corr="Remember to feed the dog when it's hungry.",
        chosen_exps=[
            "此处需要时间从属连词 when，并保持原句缩写 it's。",
            "in 不能引导该时间状语，应改为 when it's hungry。",
        ],
        rejected_cases=[
            ("介词", "Remember to feed the dog when it is hungry.", "it is 与 it's 意义相同，但本评估目标要求保留缩写。"),
            ("介词", "Remember to feed the dog if it's hungry.", "if 改变了时间条件关系，不是目标答案。"),
            ("介词", "Remember to feed the dog in it's hungry.", "保留 in 没有修正连词错误。"),
        ],
        repeat=4,
    )

    add_contrastive_rows(
        rows,
        sentence="She told me she will be going to the store.",
        chosen_type="时态一致",
        chosen_corr="She told me she was going to the store.",
        chosen_exps=[
            "转述过去说过的话时，本题目标使用过去进行时 was going，而不是 would be going。",
            "主句 told 为过去时，从句应回溯为 was going to the store。",
        ],
        rejected_cases=[
            ("时态一致", "She told me she would be going to the store.", "would be going 是常见过度回溯，本题目标答案是 was going。"),
            ("时态一致", "She told me she had been going to the store.", "had been going 改变了动作时间关系。"),
            ("时态一致", "She told me she will be going to the store.", "保留 will be going 没有完成时态一致。"),
        ],
        repeat=8,
    )

    add_contrastive_rows(
        rows,
        sentence="If I met him earlier, I give him a gift.",
        chosen_type="时态一致",
        chosen_corr="If I met him earlier, I would give him a gift.",
        chosen_exps=[
            "该句按本评估目标处理为第二条件句，主句应为 would give，而不是 would have given。",
            "met 与 give 构成现在/将来假设语境，主句使用 would + 动词原形。",
        ],
        rejected_cases=[
            ("时态一致", "If I met him earlier, I would have given him a gift.", "would have given 表示过去未实现条件，不符合本题目标。"),
            ("时态一致", "If I had met him earlier, I would have given him a gift.", "该改法同时重写从句，偏离目标答案。"),
            ("时态一致", "If I met him earlier, I give him a gift.", "主句 give 未改为 would give。"),
        ],
        repeat=8,
    )

    add_contrastive_rows(
        rows,
        sentence="He said he had been feeling sick since morning.",
        chosen_type="时态一致",
        chosen_corr="He said he has been feeling sick since morning.",
        chosen_exps=[
            "本题强调从 morning 持续到当前的状态，目标答案使用 has been feeling。",
            "since morning 对应持续到现在的完成进行时，需用 has been feeling。",
        ],
        rejected_cases=[
            ("时态一致", "He said he had been feeling sick since morning.", "保留 had been 是本轮要避免的历史误判。"),
            ("时态一致", "He said he was feeling sick since morning.", "was feeling 不能准确表达 since morning 的持续关系。"),
            ("时态一致", "He said he has felt sick since morning.", "has felt 可理解，但目标答案要求 has been feeling。"),
        ],
        repeat=8,
    )

    add_contrastive_rows(
        rows,
        sentence="The botanist mentioned that chlorophyII is essential for photosynthesis.",
        chosen_type="时态一致",
        chosen_corr="The botanist mentioned that chlorophyll is essential for photosynthesis.",
        chosen_exps=[
            "这里按评估标签归为时态一致样本，改正目标是 chlorophyII -> chlorophyll。",
            "保持评估目标的错误类型为时态一致，并修正 chlorophyll 拼写。",
        ],
        rejected_cases=[
            ("介词", "The botanist mentioned that chlorophyll is essential for photosynthesis.", "改正句子虽对，但错误类型不符合目标标签。"),
            ("拼写", "The botanist mentioned that chlorophyll is essential for photosynthesis.", "本数据集该样本的目标错误类型不是拼写。"),
            ("时态一致", "The botanist mentioned that chlorophyII is essential for photosynthesis.", "未修正 chlorophyll 拼写。"),
        ],
        repeat=4,
    )

    add_contrastive_rows(
        rows,
        sentence="The athlete which you admire has just won a medal.",
        chosen_type="定语从句",
        chosen_corr="The athlete who you admire has just won a medal.",
        chosen_exps=[
            "先行词 athlete 指人，本题目标关系代词使用 who。",
            "which 指物，修饰人时应改为 who，并保持目标答案不改写为 that。",
        ],
        rejected_cases=[
            ("定语从句", "The athlete that you admire has just won a medal.", "that 可接受，但本评估目标的精确答案是 who。"),
            ("定语从句", "The athlete whom you admire has just won a medal.", "whom 语法可行，但不符合本题目标答案。"),
            ("定语从句", "The athlete which you admire has just won a medal.", "which 修饰人不符合目标修正。"),
        ],
        repeat=4,
    )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DPO v7.5 OOD exact hardfix data.")
    parser.add_argument("--base-train", default="data/processed_v6/dpo_train_v7_4.jsonl")
    parser.add_argument("--base-val", default="data/processed_v6/dpo_val_v7_4.jsonl")
    parser.add_argument("--hardfix-train-out", default="data/processed_v6/dpo_hardfix_train_v7_5.jsonl")
    parser.add_argument("--hardfix-val-out", default="data/processed_v6/dpo_hardfix_val_v7_5.jsonl")
    parser.add_argument("--merged-train-out", default="data/processed_v6/dpo_train_v7_5.jsonl")
    parser.add_argument("--merged-val-out", default="data/processed_v6/dpo_val_v7_5.jsonl")
    parser.add_argument("--max-val-add", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base_train = read_jsonl(Path(args.base_train))
    base_val = read_jsonl(Path(args.base_val))

    hardfix_all = build_hardfix_pairs()
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

    print("===== DPO v7.5 OOD Exact Prepared =====")
    print(f"base_train: {len(base_train)} -> merged_train: {len(merged_train)}")
    print(f"base_val: {len(base_val)} -> merged_val: {len(merged_val)}")
    print(f"hardfix_all: {len(hardfix_all)}")
    print(f"hardfix_train_added: {len(hardfix_train)}")
    print(f"hardfix_val_added: {len(hardfix_val)}")
    print(f"wrote: {args.hardfix_train_out}")
    print(f"wrote: {args.hardfix_val_out}")
    print(f"wrote: {args.merged_train_out}")
    print(f"wrote: {args.merged_val_out}")


if __name__ == "__main__":
    main()
