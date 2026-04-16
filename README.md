# 英语语法纠错 SFT 项目

## 1. 项目需求

### 1.1 背景
面向中文学习者的英文写作场景，经常出现主谓一致、时态、从句结构等高频语法错误。基础大模型可以纠错，但输出不稳定，且错误类型说明不够一致。

### 1.2 目标
本项目通过构建结构化 SFT 数据并进行 LoRA 微调，让模型在以下任务上更稳定：
- 输入一条有语法问题的英文句子。
- 输出统一格式结果：
  - `错误类型`
  - `改正`
  - `解释`

### 1.3 验收标准
1. 数据层：训练数据格式统一，包含 `instruction / input / output` 三字段。  
2. 训练层：可复现跑通 `数据清洗 -> 训练 -> 导出 LoRA`。  
3. 评估层：可对比 `base` 与 `LoRA`，输出纠错准确率和分类型统计。

## 2. 实现过程（端到端）

### 阶段 A：数据构建与清洗
- 原始数据来源：`data/sft_train.jsonl`
- 清洗脚本：`train/prepare_sft_data.py`
- 主要处理逻辑：
  - 解析 `output` 中的 `错误类型/改正/解释`
  - 去除无效样本（空字段、格式错误、改正前后相同、重复输入、解释过短）
  - 基于规则对部分 `error_type` 自动归一化
  - 切分训练集/验证集

输出产物：
- `data/processed/sft_train_clean.jsonl`
- `data/processed/sft_train_v1_train.jsonl`
- `data/processed/sft_train_v1_val.jsonl`

### 阶段 B：SFT 训练（Qwen + LoRA 4bit）
- 训练脚本：`train/train_sft.py`
- 训练要点：
  - 基座模型：本地 Qwen2.5-3B-Instruct
  - 量化：4bit (bitsandbytes)
  - 微调方式：LoRA（默认 `q_proj`、`v_proj`）
  - 使用 chat template 构造监督样本

输出产物：
- `outputs/sft_v1/final/`（LoRA 权重 + tokenizer）
- `outputs/sft_v1/metrics.json`

### 阶段 C：效果评估（Base vs LoRA）
- 评估脚本：`scripts/evaluation/evaluate_lora.py`
- 评估方式：
  - 同一输入分别跑 base 和 LoRA
  - 从输出中抽取纠正句与错误类型
  - 若测试集含 gold 标签，计算：
    - `base_correction_exact_rate`
    - `lora_correction_exact_rate`
    - `lora_type_exact_rate`
    - `by_type` 分类型表现

输出产物：
- `outputs/sft_v1/eval_compare.json`

## 3. 快速开始

### 3.1 环境准备
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2 一键跑主流程
1. 数据清洗与切分
```bash
python3 train/prepare_sft_data.py \
  --input data/sft_train.jsonl \
  --output-dir data/processed \
  --min-explanation-chars 8
```

2. SFT 训练
```bash
python3 train/train_sft.py \
  --train-data data/processed/sft_train_v1_train.jsonl \
  --val-data data/processed/sft_train_v1_val.jsonl \
  --output-dir outputs/sft_v1
```

3. 评估对比
```bash
python3 scripts/evaluation/evaluate_lora.py \
  --test-file data/processed/sft_train_v1_val.jsonl \
  --lora-path outputs/sft_v1/final \
  --output outputs/sft_v1/eval_compare.json
```

## 4. 项目结构

```text
English-optimizing/
├── data/
│   ├── sft_train.jsonl
│   └── processed/
├── train/
│   ├── prepare_sft_data.py
│   └── train_sft.py
├── scripts/
│   ├── data_pipeline/   # 历史/扩展数据脚本
│   ├── evaluation/      # 评估与分析脚本
│   ├── model_utils/     # 模型下载与本地测试工具
│   └── legacy/          # 历史实验脚本
├── docs/
│   └── project-structure.md
└── README.md
```

## 5. 设计取舍

- 为什么是 LoRA：降低显存占用，便于快速迭代。
- 为什么先做数据清洗：数据质量直接决定微调上限。
- 为什么做 base 对比：避免“训练了但没有收益”不可见。

## 6. 常见问题

1. 模型路径找不到：确认本地 `models/` 下存在 Qwen2.5-3B-Instruct。  
2. 显存不足：降低 `batch-size` 或提高 `grad-accum`。  
3. 评估结果异常：先检查测试集 `output` 字段是否符合标准模板。

## 7. 后续迭代建议

1. 增加更细粒度错误类型（冠词、介词、搭配）。
2. 增加人工校验集，减少自动标注偏差。
3. 增加线上推理模板约束，提升输出格式稳定性。
