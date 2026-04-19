#!/usr/bin/env python3
"""DPO training entry (Qwen + LoRA adapter continuation)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def resolve_model_path(model_path: Optional[str]) -> str:
    if model_path:
        return model_path
    candidates = [
        "./models/Qwen2.5-3B-Instruct",
        "./models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    raise FileNotFoundError(
        "No local model found. Pass --model-path or place model under ./models/."
    )


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ensure_torch_set_submodule() -> None:
    import torch
    import torch.nn as nn

    if hasattr(nn.Module, "set_submodule"):
        return

    def set_submodule(self: nn.Module, target: str, module: nn.Module, strict: bool = False) -> None:
        if target == "":
            raise ValueError("Cannot set the submodule without a target name!")
        if not isinstance(module, nn.Module):
            raise ValueError(f"`module` is not an nn.Module, found {type(module)}")
        atoms = target.split(".")
        parent: nn.Module = self if len(atoms) == 1 else self.get_submodule(".".join(atoms[:-1]))
        if strict and not hasattr(parent, atoms[-1]):
            raise AttributeError(f"{parent._get_name()} has no attribute `{atoms[-1]}`")
        if hasattr(parent, atoms[-1]):
            mod = getattr(parent, atoms[-1])
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError(f"`{atoms[-1]}` is not an nn.Module")
        setattr(parent, atoms[-1], module)

    nn.Module.set_submodule = set_submodule  # type: ignore[method-assign]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DPO on top of SFT LoRA adapter.")
    p.add_argument("--model-path", default=None, help="Local base model path.")
    p.add_argument(
        "--sft-adapter-path",
        default="result/sft_v6_3/final",
        help="Path to SFT LoRA adapter used as DPO starting point.",
    )
    p.add_argument("--train-data", default="data/processed_v6/dpo_train_v1.jsonl")
    p.add_argument("--val-data", default="data/processed_v6/dpo_val_v1.jsonl")
    p.add_argument("--output-dir", default="result/dpo_v1")
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    from datasets import Dataset
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    _ensure_torch_set_submodule()

    model_path = resolve_model_path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.sft_adapter_path).exists():
        raise FileNotFoundError(f"sft adapter not found: {args.sft_adapter_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("===== DPO Training =====")
    print(f"base_model: {model_path}")
    print(f"sft_adapter: {args.sft_adapter_path}")
    print(f"train_data: {args.train_data}")
    print(f"val_data: {args.val_data}")
    print(f"output_dir: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_base = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=bnb_config,
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        policy_base,
        args.sft_adapter_path,
        is_trainable=True,
    )

    # Reference model: same SFT model but frozen.
    ref_base = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=bnb_config,
        dtype=torch.float16,
    )
    ref_model = PeftModel.from_pretrained(
        ref_base,
        args.sft_adapter_path,
        is_trainable=False,
    )

    train_rows = read_jsonl(args.train_data)
    val_rows = read_jsonl(args.val_data)
    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        max_prompt_length=max(64, args.max_length // 2),
        max_length=args.max_length,
        beta=args.beta,
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    final_eval = trainer.evaluate()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(final_eval, f, ensure_ascii=False, indent=2)
    print(f"saved dpo adapter to: {final_dir}")
    print(f"saved metrics to: {metrics_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

