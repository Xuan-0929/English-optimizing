#!/usr/bin/env python3
"""CLI inference entry for the grammar correction delivery model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.evaluation.evaluate_lora import (
        extract_correction,
        extract_error_type,
        generate_reply,
        load_model,
        resolve_model_path,
    )
    from scripts.evaluation.exact_constraints import apply_exact_constraints
    from scripts.evaluation.type_constraints import apply_type_constraints
except ModuleNotFoundError:
    from evaluate_lora import extract_correction, extract_error_type, generate_reply, load_model, resolve_model_path
    from exact_constraints import apply_exact_constraints
    from type_constraints import apply_type_constraints


@dataclass
class CorrectionResult:
    input: str
    error_type: str
    correction: str
    explanation: str
    raw_output: str
    type_raw: str
    type_constraint_rule: Optional[str] = None
    exact_constraint_rule: Optional[str] = None


def _extract_explanation(raw_output: str) -> str:
    marker = "**解释**:"
    if marker not in raw_output:
        return ""
    return raw_output.split(marker, 1)[1].strip()


def finalize_prediction(
    sentence: str,
    raw_output: str,
    *,
    apply_type_constraint: bool = True,
    apply_exact_constraint: bool = True,
) -> CorrectionResult:
    correction = extract_correction(raw_output)
    type_raw = extract_error_type(raw_output)
    error_type = type_raw
    type_rule = None

    if apply_type_constraint:
        error_type, type_rule = apply_type_constraints(sentence, correction, type_raw)

    exact_rule = None
    if apply_exact_constraint:
        exact_correction, exact_type, exact_rule = apply_exact_constraints(sentence)
        if exact_rule:
            correction = exact_correction or correction
            error_type = exact_type or error_type

    return CorrectionResult(
        input=sentence,
        error_type=error_type,
        correction=correction,
        explanation=_extract_explanation(raw_output),
        raw_output=raw_output,
        type_raw=type_raw,
        type_constraint_rule=type_rule,
        exact_constraint_rule=exact_rule,
    )


def format_text_result(result: CorrectionResult) -> str:
    return (
        f"**错误类型**: {result.error_type}\n"
        f"**改正**: {result.correction}\n"
        f"**解释**: {result.explanation}"
    )


def _read_input_file(path: Path) -> List[str]:
    sentences: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                row = json.loads(line)
                sentence = str(row.get("input", "")).strip()
            else:
                sentence = line
            if sentence:
                sentences.append(sentence)
    return sentences


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run grammar correction inference.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sentence", help="Single sentence to correct.")
    source.add_argument("--input-file", help="Text file or JSONL file with an `input` field.")
    parser.add_argument("--output", help="Output JSONL path for batch mode.")
    parser.add_argument("--base-model-path", default=None, help="Local base model path.")
    parser.add_argument("--lora-path", default="outputs/dpo_v7_4/final", help="LoRA adapter path.")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--raw-output", help="Bypass model generation and post-process this raw model output.")
    parser.add_argument("--json", action="store_true", help="Print JSON for single-sentence mode.")
    parser.add_argument("--no-type-constraints", action="store_true", help="Disable type constraints.")
    parser.add_argument("--no-exact-constraints", action="store_true", help="Disable exact correction constraints.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    apply_type_constraint = not args.no_type_constraints
    apply_exact_constraint = not args.no_exact_constraints

    if args.sentence:
        sentences = [args.sentence]
    else:
        sentences = _read_input_file(Path(args.input_file))

    model = tokenizer = None
    if args.raw_output is None:
        base_model_path = resolve_model_path(args.base_model_path)
        model, tokenizer = load_model(base_model_path, lora_path=args.lora_path)

    results: List[CorrectionResult] = []
    for sentence in sentences:
        if args.raw_output is not None:
            raw_output = args.raw_output
        else:
            raw_output = generate_reply(model, tokenizer, sentence, args.max_new_tokens)
        results.append(
            finalize_prediction(
                sentence,
                raw_output,
                apply_type_constraint=apply_type_constraint,
                apply_exact_constraint=apply_exact_constraint,
            )
        )

    if args.output:
        _write_jsonl(Path(args.output), (asdict(result) for result in results))
        print(f"saved: {args.output}")
        return

    if len(results) == 1:
        result = results[0]
        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False))
        else:
            print(format_text_result(result))
        return

    for result in results:
        print(json.dumps(asdict(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
