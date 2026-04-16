#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare base Qwen vs base + SFT LoRA."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_INSTRUCTION = "纠正以下句子的语法错误"
DEFAULT_TEST_SENTENCES = [
    "He said he goes to the store yesterday.",
    "The book which you recommended it is great.",
    "She don't like apples.",
    "I have been to the park last week.",
    "They is playing football now.",
    "My sister go to school every day.",
    "We will meeting tomorrow.",
    "Neither of the solutions are correct.",
]


def _try_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_user_content(sentence: str, instruction: str) -> str:
    return f"{instruction}\n{sentence}".strip()


def load_jsonl_inputs(path: str, k: int, seed: int) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            inp = obj.get("input")
            if isinstance(inp, str) and inp.strip():
                lines.append(inp.strip())
    if not lines:
        return []
    rng = random.Random(seed)
    if k > 0 and len(lines) > k:
        lines = rng.sample(lines, k)
    return lines


def load_text_inputs(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def extract_assistant_text(decoded: str) -> str:
    if "assistant\n" in decoded:
        return decoded.split("assistant\n", 1)[1].strip()
    if "assistant" in decoded:
        parts = re.split(r"assistant\s*\n?", decoded, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1:
            return parts[1].strip()
    return decoded.strip()


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    sentence: str,
    instruction: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> str:
    messages = [{"role": "user", "content": build_user_content(sentence, instruction)}]
    tmpl = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    # transformers>=4.44 常返回 BatchEncoding，不能直接 .to(device)
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
    ap = argparse.ArgumentParser(description="Compare base vs SFT LoRA")
    ap.add_argument("--base-model", default="./models/Qwen2.5-3B-Instruct")
    ap.add_argument("--lora", default="./sft_model", help="LoRA 目录（你当前仓库里多为 ./sft_model）")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--from-jsonl", default=None)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from-txt", default=None)
    ap.add_argument("-o", "--output", default="compare_base_vs_lora_results.json")
    args = ap.parse_args()

    if not os.path.isdir(args.base_model):
        raise SystemExit(f"Missing base model dir: {args.base_model}")
    if not os.path.isdir(args.lora):
        raise SystemExit(f"Missing LoRA dir: {args.lora}")

    sentences: List[str] = []
    if args.from_jsonl:
        sentences.extend(load_jsonl_inputs(args.from_jsonl, args.sample, args.seed))
    if args.from_txt:
        sentences.extend(load_text_inputs(args.from_txt))
    if not sentences:
        sentences = list(DEFAULT_TEST_SENTENCES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: no CUDA.", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if args.fp16:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16,
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
            args.base_model,
            device_map="auto" if device.type == "cuda" else None,
            quantization_config=bnb,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

    model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    results = []
    print(f"base={args.base_model}\nlora={args.lora}\ncount={len(sentences)}\n")
    for i, sent in enumerate(sentences, 1):
        print("=" * 72)
        print(f"[{i}/{len(sentences)}] {sent}")
        with model.disable_adapter():
            base_text = generate_one(
                model,
                tokenizer,
                sent,
                args.instruction,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                device,
            )
        lora_text = generate_one(
            model,
            tokenizer,
            sent,
            args.instruction,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            device,
        )
        print("--- base ---")
        print(base_text)
        print("--- lora ---")
        print(lora_text)
        results.append({"index": i, "input_sentence": sent, "base_model": base_text, "sft_lora": lora_text})

    payload = {
        "meta": {
            "base_model": args.base_model,
            "lora": args.lora,
            "fp16": args.fp16,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "instruction": args.instruction,
        },
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
