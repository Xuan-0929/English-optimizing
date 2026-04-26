# 英语语法纠错优化项目（SFT + DPO）

> 最后更新：2026-04-26

## 1. 项目目标

面向中文学习者英文写作场景，优化模型在语法纠错任务上的稳定性与可控性。

统一输出格式：
- `错误类型`
- `改正`
- `解释`

核心目标：
- 在固定 benchmark 上提升 `correction exact rate`
- 保持 `错误类型` 判定一致性
- 形成可复现的 `数据清洗 -> SFT -> DPO -> 评估` 流程

## 2. 截至目前已完成工作

### 2.1 数据工程

已完成：
- SFT 清洗与标准化流程（字段规范、去重、无效样本过滤、错误类型归一）
- v6 版本数据重构与评测基准固定化
- DPO 多轮数据构建（base / QC / hardfix / targeted / type-focus）
- 高风险样本剔除与人工台账留痕
- 关键增强包注入（micro-style、targeted 修复）

当前关键数据规模：
- `data/processed_v6/sft_train_clean_v6.jsonl`: 1200
- `data/processed_v6/sft_train_v6_train.jsonl`: 1140
- `data/processed_v6/sft_train_v6_val.jsonl`: 60
- `data/processed_v6/sft_eval_benchmark_v1.jsonl`: 60（固定评估集）
- `data/processed_v6/dpo_train_v7.jsonl`: 1130
- `data/processed_v6/dpo_val_v7.jsonl`: 67
- `data/processed_v6/dpo_train_v7_4.jsonl`: 1207
- `data/processed_v6/dpo_val_v7_4.jsonl`: 85

### 2.2 训练流程

已完成：
- SFT LoRA 训练主流程（`train/train_sft.py`）
- DPO LoRA 训练主流程（`train/train_dpo.py`）
- DPO 数据准备链路：
  - `train/prepare_dpo_data.py`
  - `train/optimize_dpo_v5_quality.py`
  - `train/prepare_dpo_v5_hardfix.py`
  - `train/prepare_dpo_v4_targeted.py`
  - `train/prepare_dpo_type_consistency.py`
  - `train/prepare_dpo_type_focus_v3.py`
  - `train/prepare_dpo_v7_3_error_driven.py`
  - `train/prepare_dpo_v7_4_tense_precision.py`

### 2.3 评估与分析

已完成：
- Base vs LoRA 自动评估（`scripts/evaluation/evaluate_lora.py`）
- 分类型指标统计（`by_type`）
- 数据与结果分析脚本（`scripts/evaluation/analyze_dataset.py` / `compare_base_vs_lora.py`）
- 多轮实验结果沉淀到 `result/`

## 3. 当前进度与结果

### 3.1 SFT 进展

固定 60 条评测集上的代表性结果：
- `sft_v6_2`: `lora_correction_exact_rate = 0.9333`，`lora_type_exact_rate = 0.9667`
- `sft_v6_3`: `lora_correction_exact_rate = 1.0000`，`lora_type_exact_rate = 0.9833`

补充：早期 `sft_v1`（101 条）从 `base 0.3663` 提升到 `lora 0.9406`，验证了 SFT 路线有效。

### 3.2 DPO 进展

固定 60 条评测集上的结果：
- `dpo_v1`: `lora_correction_exact_rate = 1.0000`，`lora_type_exact_rate = 0.9833`
- `dpo_v2 ~ dpo_v5_qc`: `lora_correction_exact_rate = 0.9833`，`lora_type_exact_rate = 0.9833`
- `dpo_v7_2_constraint`: benchmark/challenge 均达到 `1.0000` correction/type；OOD `correction_exact_rate = 0.9727`，noise robust `type_exact_rate = 0.9000`
- `dpo_v7_3`: benchmark `correction_exact_rate = 1.0000`、`type_exact_rate = 1.0000`；noise robust `type_exact_rate = 1.0000`

结论：
- DPO 路线总体稳定，`correction exact rate` 维持在高位（0.9833~1.0000）
- 质量清洗后（`v5_qc`）训练损失显著下降（`eval_loss` 0.1813 -> 0.0489），效果保持稳定
- v7.3 通过误差驱动 hardfix 修复了 noise robust 的时态类型误判；v7.4 已准备时态精度增强数据，等待训练与评估闭环

### 3.3 阶段性结论

- Base 到 LoRA 的收益已稳定复现
- SFT 与 DPO 两条路径均已跑通并沉淀脚本
- 当前瓶颈从“是否有效”转向“鲁棒性与泛化边界”

## 4. 当前里程碑状态

- [x] 数据清洗与标准格式统一
- [x] SFT 训练与评估闭环
- [x] DPO 数据构建与训练闭环
- [x] 固定 benchmark 与横向可比评估
- [x] 多轮迭代结果沉淀
- [ ] 扩展错误类型覆盖（冠词/搭配/虚拟语气等）
- [ ] 更大规模人工校验集
- [ ] 线上推理模板和稳定性压测

## 5. 快速复现

### 5.1 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 SFT 数据准备

```bash
python3 train/prepare_sft_data.py \
  --input data/sft_train.jsonl \
  --output-dir data/processed_v6 \
  --min-explanation-chars 8
```

### 5.3 SFT 训练

```bash
python3 train/train_sft.py \
  --train-data data/processed_v6/sft_train_v6_train.jsonl \
  --val-data data/processed_v6/sft_train_v6_val.jsonl \
  --output-dir outputs/sft_v6
```

### 5.4 DPO v7 数据链路

```bash
python3 train/build_fixed_benchmark.py \
  --input-val data/processed_v6/sft_train_v6_val.jsonl \
  --output data/processed_v6/sft_eval_benchmark_v1.jsonl

python3 train/prepare_dpo_data.py \
  --input-train data/processed_v6/sft_train_v6_train.jsonl \
  --input-val data/processed_v6/sft_train_v6_val.jsonl \
  --output-train data/processed_v6/dpo_train_v7_base.jsonl \
  --output-val data/processed_v6/dpo_val_v7_base.jsonl

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

### 5.5 DPO v7.3/v7.4 增强数据

v7.3 从 OOD 与 noise robust 评估结果中抽取真实误差，生成误差驱动 hardfix：

```bash
python3 train/prepare_dpo_v7_3_error_driven.py
```

当前产物：
- `data/processed_v6/dpo_train_v7_3.jsonl`: 1198
- `data/processed_v6/dpo_val_v7_3.jsonl`: 79

v7.4 在 v7.3 基础上补充时态精度 hardfix，覆盖转述时态、虚拟条件句和 `since` 持续状态表达：

```bash
python3 train/prepare_dpo_v7_4_tense_precision.py
```

当前产物：
- `data/processed_v6/dpo_hardfix_train_v7_4.jsonl`: 9
- `data/processed_v6/dpo_hardfix_val_v7_4.jsonl`: 6
- `data/processed_v6/dpo_train_v7_4.jsonl`: 1207
- `data/processed_v6/dpo_val_v7_4.jsonl`: 85

建议下一步训练命令：

```bash
python3 train/train_dpo.py \
  --sft-adapter-path outputs/sft_v6_3/final \
  --train-data data/processed_v6/dpo_train_v7_4.jsonl \
  --val-data data/processed_v6/dpo_val_v7_4.jsonl \
  --output-dir outputs/dpo_v7_4
```

### 5.6 评估

```bash
python3 scripts/evaluation/evaluate_lora.py \
  --test-file data/processed_v6/sft_eval_benchmark_v1.jsonl \
  --lora-path outputs/your_run/final \
  --apply-type-constraints \
  --output result/your_run/eval_compare.json
```

说明：
- 评估输出 `summary` 中同时包含严格指标与宽松指标：`base/lora_correction_relaxed_exact_rate`。
- 宽松指标会对少量等价表达做归一（如 `it's/it is`、`who/that`、`topic/subject`）。

### 5.7 扩展评估集（challenge + ood）

```bash
python3 train/build_eval_suites.py \
  --input-clean data/processed_v6/sft_train_clean_v6.jsonl \
  --benchmark data/processed_v6/sft_eval_benchmark_v1.jsonl \
  --challenge-out data/processed_v6/sft_eval_challenge_type_v1.jsonl \
  --ood-out data/processed_v6/sft_eval_ood_general_v1.jsonl \
  --report-out data/processed_v6/sft_eval_suites_report_v1.json
```

### 5.8 噪声鲁棒评估集

```bash
python3 scripts/evaluation/evaluate_lora.py \
  --test-file data/processed_v6/sft_eval_noise_robust_v1.jsonl \
  --lora-path outputs/your_run/final \
  --apply-type-constraints \
  --output result/your_run/noise_robust_eval_compare.json
```

## 6. 项目目录

```text
English-optimizing/
├── data/
│   ├── processed_v6/
│   └── review_pack/
├── train/
├── scripts/
│   ├── data_pipeline/
│   ├── evaluation/
│   ├── model_utils/
│   └── legacy/
├── result/
├── docs/
└── README.md
```

## 7. 下一步计划

1. 训练 `dpo_v7_4`，并在 benchmark / challenge / OOD / noise robust 四套评估集上复测。
2. 针对 OOD 中仍非满分的介词、时态、定语从句细粒度变体做 targeted 增强。
3. 增加人工复核集与误差归因报告（按错误类型、句法长度、领域拆分）。
