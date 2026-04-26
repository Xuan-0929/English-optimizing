# Grammar Correction Inference CLI

This is the minimal delivery interface for `dpo_delivery_v1`.

## Recommended Command

Run one sentence:

```bash
python3 scripts/inference/grammar_correct.py \
  --sentence "She told me she will be going to the store." \
  --lora-path outputs/dpo_v7_4/final
```

JSON output:

```bash
python3 scripts/inference/grammar_correct.py \
  --sentence "She told me she will be going to the store." \
  --lora-path outputs/dpo_v7_4/final \
  --json
```

Batch mode accepts either plain text, one sentence per line, or JSONL with an `input` field:

```bash
python3 scripts/inference/grammar_correct.py \
  --input-file input.jsonl \
  --output predictions.jsonl \
  --lora-path outputs/dpo_v7_4/final
```

## Constraints

The CLI enables both constraint layers by default:

- type constraints from `scripts/evaluation/type_constraints.py`
- exact constraints from `scripts/evaluation/exact_constraints.py`

Disable them only for raw model diagnostics:

```bash
python3 scripts/inference/grammar_correct.py \
  --sentence "..." \
  --no-type-constraints \
  --no-exact-constraints
```

## Output Contract

The JSON output contains:

- `input`
- `error_type`
- `correction`
- `explanation`
- `raw_output`
- `type_raw`
- `type_constraint_rule`
- `exact_constraint_rule`

The text output preserves the project standard format:

```text
**错误类型**: ...
**改正**: ...
**解释**: ...
```
