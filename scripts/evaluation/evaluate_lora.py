#!/usr/bin/env python3
"""Compare base vs LoRA outputs and compute metrics when gold labels exist."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from scripts.evaluation.type_constraints import apply_type_constraints
except ModuleNotFoundError:
    from type_constraints import apply_type_constraints

DEFAULT_TEST_CASES = [
    "He said he goes to the store yesterday.",
    "The book which you recommended it is great.",
    "She don't like apples.",
    "I have been to the park last week.",
    "They is playing football now.",
    "My sister go to school every day.",
    "We will meeting tomorrow.",
]

ERROR_TYPE_RE = re.compile(r"\*\*错误类型\*\*:\s*(.+?)\n")
CORRECTION_RE = re.compile(r"\*\*改正\*\*:\s*(.+?)\n")
EXPLANATION_RE = re.compile(r"\*\*解释\*\*:\s*(.+)$", re.DOTALL)
QUOTE_SENT_RE = re.compile(r'"([^"]+)"')
WS_RE = re.compile(r"\s+")
PUNCT_WS_RE = re.compile(r"\s+([,.;:!?])")


RELAXED_TOKEN_MAP = [
    (re.compile(r"\bit's\b"), "it is"),
    (re.compile(r"\bwho\b"), "<rel_pron>"),
    (re.compile(r"\bthat\b"), "<rel_pron>"),
    (re.compile(r"\btopic\b"), "subject"),
]


def _ensure_torch_set_submodule() -> None:
    import torch.nn as nn

    if hasattr(nn.Module, "set_submodule"):
        return

    def set_submodule(self, target, module, strict=False):
        if target == "":
            raise ValueError("target cannot be empty")
        atoms = target.split(".")
        if len(atoms) == 1:
            parent = self
        else:
            parent = self.get_submodule(".".join(atoms[:-1]))
        if strict and not hasattr(parent, atoms[-1]):
            raise AttributeError(f"missing submodule {atoms[-1]}")
        setattr(parent, atoms[-1], module)

    nn.Module.set_submodule = set_submodule  # type: ignore[method-assign]


def resolve_model_path(model_path: Optional[str]) -> str:
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


def load_model(base_model_path: str, lora_path: Optional[str] = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    _ensure_torch_set_submodule()
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
        dtype=torch.float16,
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


def generate_reply(model, tokenizer, sentence: str, max_new_tokens: int) -> str:
    import torch

    prompt = f"纠正以下句子的语法错误\n{sentence}"
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )
    gen = output[0][input_ids.shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def parse_sft_output(output: str) -> Optional[Tuple[str, str, str]]:
    m1 = ERROR_TYPE_RE.search(output)
    m2 = CORRECTION_RE.search(output)
    m3 = EXPLANATION_RE.search(output)
    if not (m1 and m2 and m3):
        return None
    return m1.group(1).strip(), m2.group(1).strip(), m3.group(1).strip()


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = WS_RE.sub(" ", s)
    return s.strip(" .;:")


def normalize_text_relaxed(s: str) -> str:
    s = normalize_text(s)
    for pat, rep in RELAXED_TOKEN_MAP:
        s = pat.sub(rep, s)
    s = PUNCT_WS_RE.sub(r"\1", s)
    s = WS_RE.sub(" ", s)
    return s.strip(" .;:")


def extract_correction(raw_output: str) -> str:
    parsed = parse_sft_output(raw_output)
    if parsed:
        return parsed[1]

    quoted = QUOTE_SENT_RE.findall(raw_output)
    if quoted:
        return quoted[0].strip()

    line = raw_output.splitlines()[0] if raw_output.splitlines() else raw_output
    line = line.strip().lstrip("-*0123456789. ")
    prefixes = [
        "the correct sentence is",
        "the sentence should be corrected to",
        "correct sentence:",
        "改正:",
        "正确句子是",
    ]
    low = line.lower()
    for p in prefixes:
        if low.startswith(p):
            return line[len(p) :].strip(' :"')
    return line


def extract_error_type(raw_output: str) -> str:
    parsed = parse_sft_output(raw_output)
    if parsed:
        return parsed[0]
    return ""


def load_eval_items(test_file: Optional[str]) -> List[Dict]:
    if not test_file:
        return [{"input": x, "gold_correction": None, "gold_type": None} for x in DEFAULT_TEST_CASES]

    path = Path(test_file)
    if path.suffix == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                user_input = str(row.get("input", "")).strip()
                if not user_input:
                    continue
                gold_correction = None
                gold_type = None
                output = row.get("output")
                if isinstance(output, str):
                    parsed = parse_sft_output(output)
                    if parsed:
                        gold_type, gold_correction, _ = parsed
                items.append(
                    {
                        "input": user_input,
                        "gold_correction": gold_correction,
                        "gold_type": gold_type,
                    }
                )
        return items

    with path.open("r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    return [{"input": x, "gold_correction": None, "gold_type": None} for x in lines]


def summarize_scores(items: List[Dict]) -> Dict:
    scored = [x for x in items if x.get("gold_correction")]
    if not scored:
        return {
            "scored_items": 0,
            "message": "No gold labels found in test set. Only raw comparison generated.",
        }

    total = len(scored)
    base_corr = sum(1 for x in scored if x["base_correction_exact"])
    lora_corr = sum(1 for x in scored if x["lora_correction_exact"])
    base_corr_relaxed = sum(1 for x in scored if x["base_correction_relaxed_exact"])
    lora_corr_relaxed = sum(1 for x in scored if x["lora_correction_relaxed_exact"])
    lora_type_total = sum(1 for x in scored if x.get("gold_type"))
    lora_type_ok = sum(1 for x in scored if x.get("gold_type") and x["lora_type_exact"])

    by_type = defaultdict(lambda: {"count": 0, "base_correction_exact": 0, "lora_correction_exact": 0, "lora_type_exact": 0})
    for x in scored:
        t = x.get("gold_type") or "UNKNOWN"
        by_type[t]["count"] += 1
        by_type[t]["base_correction_exact"] += int(bool(x["base_correction_exact"]))
        by_type[t]["lora_correction_exact"] += int(bool(x["lora_correction_exact"]))
        by_type[t]["lora_type_exact"] += int(bool(x["lora_type_exact"]))

    by_type_rate = {}
    for t, m in by_type.items():
        n = max(1, m["count"])
        by_type_rate[t] = {
            "count": m["count"],
            "base_correction_exact_rate": round(m["base_correction_exact"] / n, 4),
            "lora_correction_exact_rate": round(m["lora_correction_exact"] / n, 4),
            "lora_type_exact_rate": round(m["lora_type_exact"] / n, 4),
        }

    return {
        "scored_items": total,
        "base_correction_exact_rate": round(base_corr / total, 4),
        "lora_correction_exact_rate": round(lora_corr / total, 4),
        "base_correction_relaxed_exact_rate": round(base_corr_relaxed / total, 4),
        "lora_correction_relaxed_exact_rate": round(lora_corr_relaxed / total, 4),
        "lora_type_exact_rate": round(lora_type_ok / max(1, lora_type_total), 4),
        "by_type": by_type_rate,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare base vs LoRA model outputs.")
    p.add_argument("--base-model-path", default=None, help="Local base model path.")
    p.add_argument("--lora-path", default="outputs/sft_v1/final", help="LoRA adapter path.")
    p.add_argument("--test-file", default=None, help="Optional .txt or .jsonl file.")
    p.add_argument("--output", default="outputs/sft_v1/eval_compare.json", help="Output JSON path.")
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument(
        "--apply-type-constraints",
        action="store_true",
        help="Apply deterministic post-processing constraints on predicted error type.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    base_model_path = resolve_model_path(args.base_model_path)
    eval_items = load_eval_items(args.test_file)

    print("===== Evaluate Base vs LoRA =====")
    print(f"base_model_path: {base_model_path}")
    print(f"lora_path: {args.lora_path}")
    print(f"test_items: {len(eval_items)}")

    base_model, base_tok = load_model(base_model_path, lora_path=None)
    lora_model, lora_tok = load_model(base_model_path, lora_path=args.lora_path)

    results: List[Dict] = []
    for idx, item in enumerate(eval_items, start=1):
        sentence = item["input"]
        gold_correction = item.get("gold_correction")
        gold_type = item.get("gold_type")

        base_raw = generate_reply(base_model, base_tok, sentence, args.max_new_tokens)
        lora_raw = generate_reply(lora_model, lora_tok, sentence, args.max_new_tokens)
        base_corr = extract_correction(base_raw)
        lora_corr = extract_correction(lora_raw)
        lora_type_raw = extract_error_type(lora_raw)
        lora_type = lora_type_raw
        applied_rule = None
        if args.apply_type_constraints:
            lora_type, applied_rule = apply_type_constraints(sentence, lora_corr, lora_type_raw)

        row = {
            "id": idx,
            "input": sentence,
            "gold_correction": gold_correction,
            "gold_type": gold_type,
            "base_output": base_raw,
            "lora_output": lora_raw,
            "base_correction": base_corr,
            "lora_correction": lora_corr,
            "lora_type_raw": lora_type_raw,
            "lora_type": lora_type,
            "lora_type_constraint_rule": applied_rule,
            "base_correction_exact": None,
            "lora_correction_exact": None,
            "base_correction_relaxed_exact": None,
            "lora_correction_relaxed_exact": None,
            "lora_type_exact": None,
        }

        if gold_correction:
            gold_norm = normalize_text(gold_correction)
            gold_relaxed_norm = normalize_text_relaxed(gold_correction)
            row["base_correction_exact"] = normalize_text(base_corr) == gold_norm
            row["lora_correction_exact"] = normalize_text(lora_corr) == gold_norm
            row["base_correction_relaxed_exact"] = normalize_text_relaxed(base_corr) == gold_relaxed_norm
            row["lora_correction_relaxed_exact"] = normalize_text_relaxed(lora_corr) == gold_relaxed_norm
        if gold_type:
            row["lora_type_exact"] = normalize_text(lora_type) == normalize_text(gold_type)

        results.append(row)

    summary = summarize_scores(results)

    out = {
        "summary": summary,
        "items": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
