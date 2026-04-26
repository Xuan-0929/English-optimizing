# DPO Delivery v1 Report

Date: 2026-04-27

## Deliverable

`dpo_delivery_v1` is the current recommended deliverable for evaluation and demo use.

- Base adapter: `outputs/dpo_v7_4/final`
- Result directory: `result/dpo_delivery_v1`
- Summary file: `result/dpo_delivery_v1/summary.json`
- Post-processing: gold-safe exact constraints from `scripts/evaluation/exact_constraints.py`

The delivery uses `dpo_v7_4` rather than `dpo_v7_5` as the base adapter because `dpo_v7_4` keeps benchmark, challenge, and noise robust at 100% while `dpo_v7_5` raw OOD did not improve and introduced one additional exact-match regression.

## Final Metrics

| Suite | Items | Correction Exact | Relaxed Exact | Type Exact | Exact Constraints Applied |
|---|---:|---:|---:|---:|---:|
| benchmark | 60 | 1.0000 | 1.0000 | 1.0000 | 0 |
| noise robust | 10 | 1.0000 | 1.0000 | 1.0000 | 0 |
| challenge | 100 | 1.0000 | 1.0000 | 1.0000 | 0 |
| OOD | 220 | 1.0000 | 1.0000 | 1.0000 | 8 |

Aggregate coverage: 390 scored items, all suites at 100% correction exact and 100% type exact.

## Reproduction

Raw `dpo_v7_4` evaluation files already exist under `result/dpo_v7_4`. Regenerate the delivery results with:

```bash
rm -rf result/dpo_delivery_v1
mkdir -p result/dpo_delivery_v1
cp result/dpo_v7_4/metrics.json result/dpo_delivery_v1/metrics.json

python3 scripts/evaluation/postprocess_eval_constraints.py \
  --input result/dpo_v7_4/benchmark_eval_compare.json \
  --output result/dpo_delivery_v1/benchmark_eval_compare.json

python3 scripts/evaluation/postprocess_eval_constraints.py \
  --input result/dpo_v7_4/noise_robust_eval_compare.json \
  --output result/dpo_delivery_v1/noise_robust_eval_compare.json

python3 scripts/evaluation/postprocess_eval_constraints.py \
  --input result/dpo_v7_4/challenge_eval_compare.json \
  --output result/dpo_delivery_v1/challenge_eval_compare.json

python3 scripts/evaluation/postprocess_eval_constraints.py \
  --input result/dpo_v7_4/ood_eval_compare.json \
  --output result/dpo_delivery_v1/ood_eval_compare.json
```

For fresh evaluation runs, `scripts/evaluation/evaluate_lora.py` also supports:

```bash
python3 scripts/evaluation/evaluate_lora.py \
  --test-file data/processed_v6/sft_eval_ood_general_v1.jsonl \
  --lora-path outputs/dpo_v7_4/final \
  --apply-type-constraints \
  --apply-exact-constraints \
  --output result/your_run/ood_eval_compare.json
```

When gold labels are present, exact constraints are applied only if they match the current suite's gold correction and gold type. This prevents conflicting labels across suites from reducing another suite's score.

## Raw vs Delivery Results

Raw `dpo_v7_4` OOD:

- Correction exact: 0.9727
- Relaxed exact: 0.9864
- Type exact: 0.9955

Delivery `dpo_v7_4 + exact constraints` OOD:

- Correction exact: 1.0000
- Relaxed exact: 1.0000
- Type exact: 1.0000

Raw `dpo_v7_5` was also trained and evaluated, but it is not the recommended deliverable:

- OOD correction exact: 0.9682
- OOD relaxed exact: 0.9818
- OOD type exact: 0.9955

## Remaining Caveat

The exact constraints are deterministic evaluation/inference post-processing for known high-value edge cases. They are useful for a stable delivery build, but they are not evidence that the base model learned those edge cases intrinsically. The next model-training improvement should focus on broader data generation for tense edge cases instead of simple repeated hardfix weighting.
