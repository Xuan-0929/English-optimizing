# 英语语法纠错项目

## 当前主流程（推荐）

请在仓库根目录执行以下命令。

1. 数据清洗与切分
```bash
python3 train/prepare_sft_data.py \
  --input data/sft_train.jsonl \
  --output-dir data/processed \
  --min-explanation-chars 8
```

2. SFT 训练（Qwen + LoRA 4bit）
```bash
python3 train/train_sft.py \
  --train-data data/processed/sft_train_v1_train.jsonl \
  --val-data data/processed/sft_train_v1_val.jsonl \
  --output-dir outputs/sft_v1
```

3. 评估与对比（base vs LoRA）
```bash
python3 scripts/evaluation/evaluate_lora.py \
  --test-file data/processed/sft_train_v1_val.jsonl \
  --lora-path outputs/sft_v1/final \
  --output outputs/sft_v1/eval_compare.json
```

## 新项目结构

```text
English-optimizing/
├── README.md
├── requirements.txt
├── data/
│   ├── sft_train.jsonl
│   └── processed/
├── train/
│   ├── prepare_sft_data.py
│   └── train_sft.py
├── scripts/
│   ├── data_pipeline/
│   │   ├── generate_sft_data.py
│   │   ├── data_gen.py
│   │   ├── data_expansion.py
│   │   ├── filter.py
│   │   └── filter_sft.py
│   ├── evaluation/
│   │   ├── evaluate_lora.py
│   │   ├── compare_base_vs_lora.py
│   │   ├── eval_en_writing_base.py
│   │   ├── check_sft_quality.py
│   │   └── analyze_dataset.py
│   ├── model_utils/
│   │   ├── load_model.py
│   │   └── test_local_model.py
│   └── legacy/
│       ├── Req_veri_web.py
│       └── Requirement_test.py
└── docs/
    └── project-structure.md
```

## 脚本分层说明

- `train/`：当前训练主链路，优先使用。
- `scripts/data_pipeline/`：历史/扩展数据生成与过滤流程。
- `scripts/evaluation/`：评估与质量检查工具脚本。
- `scripts/model_utils/`：模型下载、本地测试等辅助脚本。
- `scripts/legacy/`：暂存旧实验脚本，不参与主流程。

## 常用命令（新路径）

下载模型：
```bash
python3 scripts/model_utils/load_model.py
```

生成直出训练数据（旧流程）：
```bash
python3 scripts/data_pipeline/generate_sft_data.py
```

过滤直出训练数据（旧流程）：
```bash
python3 scripts/data_pipeline/filter_sft.py
```

## 补充文档

详细说明见：`docs/project-structure.md`
