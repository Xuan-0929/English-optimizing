#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用本地基座模型评估「中文原意 -> 学习者英文」的表达质量。

示例:
  .venv/Scripts/python.exe eval_en_writing_base.py --zh "我想下周请假两天回老家。" --en "I want to ask two days leave next week to go back my hometown."
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _project_root() -> Path:
    """脚本所在目录视为项目根，避免依赖终端当前工作目录。"""
    return Path(__file__).resolve().parent


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def resolve_local_qwen25_3b_instruct(explicit: str | None) -> str:
    env = os.environ.get("QWEN25_BASE_MODEL", "").strip()
    if env and explicit is None:
        explicit = env

    root = _project_root()

    if explicit:
        cand = Path(explicit)
        if not cand.is_absolute():
            cand = (root / cand).resolve()
        else:
            cand = cand.resolve()
        if _is_model_dir(cand):
            return str(cand)
        raise SystemExit(f"找不到有效基座目录（需要含 config.json）: {cand}")

    direct = root / "models" / "Qwen2.5-3B-Instruct"
    if _is_model_dir(direct):
        return str(direct)

    snap_root = root / "models" / "models--Qwen--Qwen2.5-3B-Instruct" / "snapshots"
    if snap_root.is_dir():
        for h in sorted((d.name for d in snap_root.iterdir() if d.is_dir()), reverse=True):
            cand = snap_root / h
            if _is_model_dir(cand):
                return str(cand)

    raise SystemExit(
        "未找到本地 Qwen2.5-3B-Instruct。\n"
        f"已相对项目根查找: {root}\n"
        "请放到 models/Qwen2.5-3B-Instruct，或设置环境变量 QWEN25_BASE_MODEL，或传 --base-model。"
    )


def _try_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_user_prompt(zh: str, en: str) -> str:
    return (
        "你是一位英语教师，正在帮助中国学生提高书面英语。\n\n"
        f"【中文原意】\n{zh.strip()}\n\n"
        f"【学习者英文】\n{en.strip()}\n\n"
        "请根据中文原意，完成：\n"
        "1）这段英文是否准确传达原意；\n"
        "2）语法、搭配、时态、冠词等有无问题；\n"
        "3）是否自然、有无中式英语；\n"
        "4）给出修改后的推荐英文（如需要）。\n"
        "请用中文作答，分点简要说明。"
    )


def extract_assistant_text(decoded: str) -> str:
    if "assistant\n" in decoded:
        return decoded.split("assistant\n", 1)[1].strip()
    if "assistant" in decoded:
        parts = re.split(r"assistant\s*\n?", decoded, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1:
            return parts[1].strip()
    return decoded.strip()


@torch.inference_mode()
def run_eval(
    model,
    tokenizer,
    user_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> str:
    messages = [{"role": "user", "content": user_text}]
    tmpl = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    if isinstance(tmpl, torch.Tensor):
        input_ids = tmpl.to(device)
    else:
        input_ids = tmpl["input_ids"].to(device)

    do_sample = temperature > 0
    gen_kw = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kw["temperature"] = temperature
        gen_kw["top_p"] = top_p

    out = model.generate(input_ids, **gen_kw)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return extract_assistant_text(decoded)


def main() -> None:
    _try_utf8_stdout()
    ap = argparse.ArgumentParser(description="基座模型评估中英翻译/英文表达")
    ap.add_argument("--zh", required=True, help="中文原意")
    ap.add_argument("--en", required=True, help="待评估的英文（你的翻译或习作）")
    ap.add_argument(
        "--base-model",
        default=None,
        help="基座目录（含 config.json）。省略则相对脚本所在项目根自动查找；也可用环境变量 QWEN25_BASE_MODEL",
    )
    ap.add_argument("--fp16", action="store_true", help="FP16 全量加载（不用 4bit）")
    ap.add_argument("--local-only", action="store_true", help="仅使用本地文件，不联网")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    base_model = resolve_local_qwen25_3b_instruct(args.base_model)
    print(f"（基座路径）{base_model}", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("（提示）未检测到 CUDA，将用 CPU，速度较慢。", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, local_files_only=args.local_only
    )
    tokenizer.pad_token = tokenizer.eos_token

    load_kw = dict(trust_remote_code=True, local_files_only=args.local_only)
    if args.fp16:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto" if device.type == "cuda" else None,
            torch_dtype=torch.float16,
            **load_kw,
        )
        if device.type != "cuda":
            model = model.to(device)
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto" if device.type == "cuda" else None,
            quantization_config=bnb,
            torch_dtype=torch.float16,
            **load_kw,
        )

    model.eval()
    prompt = build_user_prompt(args.zh, args.en)
    text = run_eval(
        model,
        tokenizer,
        prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        device,
    )

    print("========== 输入 ==========")
    print("【中文原意】", args.zh.strip())
    print("【学习者英文】", args.en.strip())
    print("========== 基座评估 ==========")
    print(text)


if __name__ == "__main__":
    main()
