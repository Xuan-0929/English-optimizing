#!/usr/bin/env python3
"""SFT v1 training entry for Qwen + LoRA (4bit)."""

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


def read_sft_jsonl(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
    return rows


def build_tokenize_fn(tokenizer, max_length: int):
    def tokenize_fn(examples):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for instruction, user_input, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            prompt = f"{instruction}\n{user_input}".strip()
            messages_prompt = [{"role": "user", "content": prompt}]
            messages_full = messages_prompt + [{"role": "assistant", "content": output}]

            prompt_text = tokenizer.apply_chat_template(
                messages_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            full = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            full_ids = full["input_ids"]
            full_mask = full["attention_mask"]

            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            input_ids_list.append(full_ids)
            attention_mask_list.append(full_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    return tokenize_fn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Qwen SFT v1 with LoRA.")
    parser.add_argument("--model-path", default=None, help="Local base model path.")
    parser.add_argument(
        "--train-data",
        default="data/processed/sft_train_v1_train.jsonl",
        help="Train JSONL path.",
    )
    parser.add_argument(
        "--val-data",
        default="data/processed/sft_train_v1_val.jsonl",
        help="Validation JSONL path.",
    )
    parser.add_argument("--output-dir", default="outputs/sft_v1", help="Output dir.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "v_proj"],
        help="LoRA target module names.",
    )
    parser.add_argument(
        "--incremental-lora-path",
        default=None,
        help="Existing LoRA path to continue training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from datasets import Dataset
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    model_path = resolve_model_path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("===== SFT v1 Training =====")
    print(f"model_path: {model_path}")
    print(f"train_data: {args.train_data}")
    print(f"val_data: {args.val_data}")
    print(f"output_dir: {args.output_dir}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.incremental_lora_path:
        if not Path(args.incremental_lora_path).exists():
            raise FileNotFoundError(
                f"incremental_lora_path not found: {args.incremental_lora_path}"
            )
        model = PeftModel.from_pretrained(model, args.incremental_lora_path)
    else:
        for p in model.parameters():
            p.requires_grad = False
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()

    train_rows = read_sft_jsonl(args.train_data)
    val_rows = read_sft_jsonl(args.val_data)
    print(f"train records: {len(train_rows)}")
    print(f"val records: {len(val_rows)}")

    tokenize_fn = build_tokenize_fn(tokenizer, args.max_length)
    train_ds = Dataset.from_list(train_rows).map(
        tokenize_fn,
        batched=True,
        remove_columns=["instruction", "input", "output"],
    )
    val_ds = Dataset.from_list(val_rows).map(
        tokenize_fn,
        batched=True,
        remove_columns=["instruction", "input", "output"],
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100),
    )

    trainer.train()
    final_eval = trainer.evaluate()
    print(f"final_eval: {final_eval}")

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(final_eval, f, ensure_ascii=False, indent=2)
    print(f"saved model to: {final_dir}")
    print(f"saved metrics to: {metrics_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
