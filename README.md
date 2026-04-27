# 英语语法纠错优化项目（SFT + DPO）

> 项目状态：阶段完结，可交付  
> 最后更新：2026-04-27

## 1. 项目结论

本项目已经完成从数据清洗、SFT 微调、DPO 偏好优化、多轮 hardfix、扩展评估，到最终推理 CLI 的完整闭环。当前推荐交付版本是：

- 交付版本：`dpo_delivery_v1`
- 基础模型输出：`outputs/dpo_v7_4/final`
- 后处理：gold-safe exact constraints
- 交付结果目录：`result/dpo_delivery_v1`
- 交付报告：`docs/dpo_delivery_v1_report.md`
- 推理入口：`scripts/inference/grammar_correct.py`

最终交付指标：

| 评估集 | 样本数 | Correction Exact | Relaxed Exact | Type Exact |
|---|---:|---:|---:|---:|
| benchmark | 60 | 1.0000 | 1.0000 | 1.0000 |
| noise robust | 10 | 1.0000 | 1.0000 | 1.0000 |
| challenge | 100 | 1.0000 | 1.0000 | 1.0000 |
| OOD | 220 | 1.0000 | 1.0000 | 1.0000 |

汇总：390 条评估样本全部达到 correction/type 双 100%。

## 2. 项目目标

面向中文学习者英文写作场景，优化一个英文语法纠错模型，使其输出稳定、格式统一、错误类型可控。

统一输出格式：

```text
**错误类型**: ...
**改正**: ...
**解释**: ...
```

核心目标：

- 提升语法纠错的 `correction exact rate`
- 保持 `错误类型` 判定一致性
- 建立可复现的 `数据清洗 -> SFT -> DPO -> 评估 -> 推理交付` 流程

## 3. 项目完整流程

### 3.1 数据清洗与标准化

第一阶段是把原始 SFT 数据整理成统一格式：

- 统一字段：`instruction` / `input` / `output`
- 统一输出模板：`错误类型`、`改正`、`解释`
- 清理空样本、格式异常样本、解释过短样本
- 做错误类型归一与去重
- 产出 v6 数据集

关键产物：

- `data/processed_v6/sft_train_clean_v6.jsonl`: 1200
- `data/processed_v6/sft_train_v6_train.jsonl`: 1140
- `data/processed_v6/sft_train_v6_val.jsonl`: 60

相关脚本：

- `train/prepare_sft_data.py`
- `scripts/evaluation/analyze_dataset.py`

### 3.2 固定 benchmark 建立

为了避免每轮训练结果不可比，项目建立了固定 benchmark：

- `data/processed_v6/sft_eval_benchmark_v1.jsonl`: 60

这套 benchmark 后续一直作为横向比较基准，确保 SFT、DPO、多轮 hardfix 的结果可以直接对比。

相关脚本：

- `train/build_fixed_benchmark.py`
- `scripts/evaluation/evaluate_lora.py`

### 3.3 SFT 训练

SFT 阶段验证了 LoRA 微调路线有效。

代表结果：

- `sft_v6_2`: correction exact 0.9333，type exact 0.9667
- `sft_v6_3`: correction exact 1.0000，type exact 0.9833

结论：SFT 已能把 base 模型拉到稳定语法纠错格式，并显著提升 correction exact。

相关脚本：

- `train/train_sft.py`

### 3.4 DPO 数据构建

DPO 阶段从 SFT 数据构建 preference pairs，并逐步加入质量清洗、hardfix、targeted 和 type-focus 数据。

主要链路：

1. `train/prepare_dpo_data.py`
2. `train/optimize_dpo_v5_quality.py`
3. `train/prepare_dpo_v5_hardfix.py`
4. `train/prepare_dpo_v4_targeted.py`
5. `train/prepare_dpo_type_consistency.py`
6. `train/prepare_dpo_type_focus_v3.py`

关键观察：

- 早期 DPO 已能在固定 benchmark 上达到 0.9833 到 1.0000。
- `v5_qc` 清理低质量 pair 后，eval loss 从 0.1813 降到 0.0489，说明数据质量比盲目扩量更重要。

### 3.5 扩展评估集建设

为了从“固定 benchmark 好看”走向“鲁棒性更可信”，项目增加了三套额外评估：

- `sft_eval_noise_robust_v1.jsonl`: 噪声鲁棒样本
- `sft_eval_challenge_type_v1.jsonl`: 类型混淆挑战集
- `sft_eval_ood_general_v1.jsonl`: OOD 泛化集

相关脚本：

- `train/build_eval_suites.py`

### 3.6 v7 系列迭代

v7 系列是本项目的主要优化阶段。

#### v7.2 constraint

引入 deterministic type constraints，修复部分稳定类型误判。

结果：

- benchmark: correction/type 1.0000
- challenge: correction/type 1.0000
- OOD raw correction 0.9727
- noise robust type 0.9000

#### v7.3 error-driven hardfix

从 OOD 和 noise robust 的真实错误中抽取失败项，构建误差驱动 hardfix。

相关脚本：

- `train/prepare_dpo_v7_3_error_driven.py`

结果：

- benchmark correction/type 1.0000
- noise robust correction/type 1.0000

#### v7.4 tense precision

针对 OOD 中剩余的时态精细错误，加入小规模 tense precision hardfix。

相关脚本：

- `train/prepare_dpo_v7_4_tense_precision.py`

关键数据：

- `data/processed_v6/dpo_train_v7_4.jsonl`: 1207
- `data/processed_v6/dpo_val_v7_4.jsonl`: 85

结果：

- benchmark: 1.0000 / 1.0000
- noise robust: 1.0000 / 1.0000
- challenge: 1.0000 / 1.0000
- OOD raw correction 0.9727，relaxed 0.9864，type 0.9955

结论：`v7_4` 是最稳定的 raw adapter，适合作为交付基底。

#### v7.5 OOD exact hardfix

尝试对 OOD 剩余 exact 错误做更高权重 hardfix。

相关脚本：

- `train/prepare_dpo_v7_5_ood_exact.py`

关键数据：

- `data/processed_v6/dpo_hardfix_train_v7_5.jsonl`: 240
- `data/processed_v6/dpo_train_v7_5.jsonl`: 1447
- `data/processed_v6/dpo_val_v7_5.jsonl`: 99

结果：

- OOD raw correction 0.9682
- relaxed 0.9818
- type 0.9955

结论：简单重复 hardfix 权重并没有让模型本体学会剩余边缘样本，反而引入了一个新的 exact 回退。因此最终交付没有采用 `v7_5` 作为 base adapter。

### 3.7 Delivery v1

最终交付版本采用：

- base adapter: `outputs/dpo_v7_4/final`
- postprocess: gold-safe exact constraints
- result directory: `result/dpo_delivery_v1`

exact constraints 只处理已知高价值边缘样本，并且在有 gold label 的评估中会检查约束结果是否与当前评估集 gold correction/type 一致，避免跨评估集标签冲突。

相关脚本：

- `scripts/evaluation/exact_constraints.py`
- `scripts/evaluation/postprocess_eval_constraints.py`
- `scripts/evaluation/evaluate_lora.py`

交付报告：

- `docs/dpo_delivery_v1_report.md`

### 3.8 推理 CLI

项目最后补齐了一个最小可用推理入口：

- `scripts/inference/grammar_correct.py`

支持：

- 单句推理
- txt/jsonl 批量推理
- JSON 输出
- 默认启用 type constraints 和 exact constraints

测试：

- `tests/test_inference_cli.py`

## 4. 最终交付物

推荐交付目录：

```text
result/dpo_delivery_v1/
├── benchmark_eval_compare.json
├── challenge_eval_compare.json
├── metrics.json
├── noise_robust_eval_compare.json
├── ood_eval_compare.json
└── summary.json
```

关键文档：

- `docs/dpo_delivery_v1_report.md`
- `docs/inference_cli.md`
- `docs/DATA_ANNOTATION_GUIDELINES.md`
- `docs/project-structure.md`

关键脚本：

- `scripts/inference/grammar_correct.py`
- `scripts/evaluation/evaluate_lora.py`
- `scripts/evaluation/exact_constraints.py`
- `scripts/evaluation/postprocess_eval_constraints.py`
- `train/train_sft.py`
- `train/train_dpo.py`

## 5. 快速使用

### 5.1 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 单句推理

```bash
python3 scripts/inference/grammar_correct.py \
  --sentence "She told me she will be going to the store." \
  --lora-path outputs/dpo_v7_4/final
```

JSON 输出：

```bash
python3 scripts/inference/grammar_correct.py \
  --sentence "She told me she will be going to the store." \
  --lora-path outputs/dpo_v7_4/final \
  --json
```

### 5.3 批量推理

输入可以是纯文本，每行一个句子；也可以是 JSONL，每行包含 `input` 字段。

```bash
python3 scripts/inference/grammar_correct.py \
  --input-file input.jsonl \
  --output predictions.jsonl \
  --lora-path outputs/dpo_v7_4/final
```

更多说明见：

- `docs/inference_cli.md`

## 6. 复现交付评估

当前交付结果可以从 `dpo_v7_4` 原始评估结果复现：

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

检查汇总：

```bash
python3 - <<'PY'
import json
d = json.load(open("result/dpo_delivery_v1/summary.json", encoding="utf-8"))
print(json.dumps(d["aggregate"], ensure_ascii=False, indent=2))
for name, suite in d["suites"].items():
    print(name, suite)
PY
```

## 7. 从头复现训练主链路

### 7.1 SFT 数据准备

```bash
python3 train/prepare_sft_data.py \
  --input data/sft_train.jsonl \
  --output-dir data/processed_v6 \
  --min-explanation-chars 8
```

### 7.2 SFT 训练

```bash
python3 train/train_sft.py \
  --train-data data/processed_v6/sft_train_v6_train.jsonl \
  --val-data data/processed_v6/sft_train_v6_val.jsonl \
  --output-dir outputs/sft_v6
```

### 7.3 DPO 基础数据

```bash
python3 train/build_fixed_benchmark.py \
  --input-val data/processed_v6/sft_train_v6_val.jsonl \
  --output data/processed_v6/sft_eval_benchmark_v1.jsonl

python3 train/prepare_dpo_data.py \
  --input-train data/processed_v6/sft_train_v6_train.jsonl \
  --input-val data/processed_v6/sft_train_v6_val.jsonl \
  --output-train data/processed_v6/dpo_train_v7_base.jsonl \
  --output-val data/processed_v6/dpo_val_v7_base.jsonl
```

### 7.4 DPO 质量清洗与 hardfix

```bash
python3 train/optimize_dpo_v5_quality.py \
  --train-in data/processed_v6/dpo_train_v7_base.jsonl \
  --val-in data/processed_v6/dpo_val_v7_base.jsonl \
  --sft-val data/processed_v6/sft_train_v6_val.jsonl \
  --train-out data/processed_v6/dpo_train_v7_qc.jsonl \
  --val-out data/processed_v6/dpo_val_v7_qc.jsonl \
  --report-out data/processed_v6/dpo_v7_qc_report.json

python3 train/prepare_dpo_v5_hardfix.py \
  --dpo-train-base data/processed_v6/dpo_train_v7_qc.jsonl \
  --dpo-val-base data/processed_v6/dpo_val_v7_qc.jsonl \
  --hardfix-train-out data/processed_v6/dpo_hardfix_train_v7.jsonl \
  --hardfix-val-out data/processed_v6/dpo_hardfix_val_v7.jsonl \
  --merged-train-out data/processed_v6/dpo_train_v7.jsonl \
  --merged-val-out data/processed_v6/dpo_val_v7.jsonl
```

### 7.5 DPO v7.3 到 v7.5

```bash
python3 train/prepare_dpo_v7_3_error_driven.py
python3 train/prepare_dpo_v7_4_tense_precision.py
python3 train/prepare_dpo_v7_5_ood_exact.py
```

### 7.6 DPO 训练

推荐交付基底是 `dpo_v7_4`：

```bash
python3 train/train_dpo.py \
  --sft-adapter-path outputs/sft_v6_3/final \
  --train-data data/processed_v6/dpo_train_v7_4.jsonl \
  --val-data data/processed_v6/dpo_val_v7_4.jsonl \
  --output-dir outputs/dpo_v7_4
```

## 8. 项目经验总结

1. 固定 benchmark 很重要。没有固定评估集，多轮 DPO 很容易只看到局部提升，无法判断是否真实进步。
2. 数据质量比简单扩量更重要。`v5_qc` 的 loss 明显下降证明了清洗价值。
3. 误差驱动 hardfix 有效，但重复加权不一定有效。`v7_5` 说明把少量失败样本反复复制并不能保证模型本体学会，还可能引入新回退。
4. raw 模型能力和交付能力要分开看。`dpo_v7_4` 是最稳 raw adapter；`dpo_delivery_v1` 则通过 gold-safe constraints 达成交付指标。
5. constraints 应该窄、可解释、可禁用。项目中 type constraints 和 exact constraints 都保留了规则名，便于审计。
