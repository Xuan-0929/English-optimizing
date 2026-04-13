#!/usr/bin/env python3
"""Compare base model vs LoRA model on grammar-correction prompts."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

DEFAULT_TEST_CASES = [
    "He said he goes to the store yesterday.",
    "The book which you recommended it is great.",
    "She don't like apples.",
    "I have been to the park last week.",
    "They is playing football now.",
    "My sister go to school every day.",
    "We will meeting tomorrow.",
]


def resolve_model_path(model_path: str = None) -> str:
    if model_path:
        return model_path
    candidates = [
        "./models/Qwen2.5-3B-Instruct",
        "./models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError("No local base model found. Please pass --base-model-path.")


def load_model(base_model_path: str, lora_path: str = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, sentence: str, max_new_tokens: int = 160) -> str:
    import torch

    prompt = f"纠正以下句子的语法错误\n{sentence}"
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.05,
        )
    generated = output[0][input_ids.shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_test_cases(test_file: str = None) -> List[str]:
    if not test_file:
        return DEFAULT_TEST_CASES
    path = Path(test_file)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "input" in item:
                    rows.append(str(item["input"]).strip())
        return rows
    with path.open("r", encoding="utf-8") as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def build_parser():
    parser = argparse.ArgumentParser(description="Compare base vs LoRA model outputs.")
    parser.add_argument("--base-model-path", default=None, help="Local base model path.")
    parser.add_argument(
        "--lora-path",
        default="outputs/sft_v1/final",
        help="LoRA adapter path.",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        help="Optional test file (.txt or .jsonl with input field).",
    )
    parser.add_argument(
        "--output",
        default="outputs/sft_v1/eval_compare.json",
        help="Output JSON path.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_model_path = resolve_model_path(args.base_model_path)
    test_cases = load_test_cases(args.test_file)

    print("===== Evaluate Base vs LoRA =====")
    print(f"base_model_path: {base_model_path}")
    print(f"lora_path: {args.lora_path}")
    print(f"test_cases: {len(test_cases)}")

    base_model, base_tok = load_model(base_model_path, lora_path=None)
    lora_model, lora_tok = load_model(base_model_path, lora_path=args.lora_path)

    results: List[Dict] = []
    for idx, sentence in enumerate(test_cases, start=1):
        base_out = generate_reply(base_model, base_tok, sentence, args.max_new_tokens)
        lora_out = generate_reply(lora_model, lora_tok, sentence, args.max_new_tokens)
        result = {
            "id": idx,
            "input": sentence,
            "base_output": base_out,
            "lora_output": lora_out,
        }
        results.append(result)
        print(f"[{idx}] input: {sentence}")
        print(f"  base: {base_out}")
        print(f"  lora: {lora_out}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
