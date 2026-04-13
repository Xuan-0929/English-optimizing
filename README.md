# 英语语法纠错项目 - 存储路径与维护指南

## SFT v1 快速开始（当前可直接运行）

下面是当前仓库可直接跑通的 SFT v1 流程。

1. 准备清洗并切分训练数据
```bash
python3 train/prepare_sft_data.py \
  --input data/sft_train.jsonl \
  --output-dir data/processed
```

2. 运行 SFT 训练（Qwen + LoRA 4bit）
```bash
python3 train/train_sft.py \
  --train-data data/processed/sft_train_v1_train.jsonl \
  --val-data data/processed/sft_train_v1_val.jsonl \
  --output-dir outputs/sft_v1
```

3. 对比 base 与 LoRA 输出
```bash
python3 evaluate_lora.py \
  --lora-path outputs/sft_v1/final \
  --output outputs/sft_v1/eval_compare.json
```

说明：
- `train/train_sft.py` 会自动尝试本地模型路径：
  - `./models/Qwen2.5-3B-Instruct`
  - `./models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1`
- 如你的模型在其他目录，手动传 `--model-path <path>` 即可。
- 训练产物默认在 `outputs/sft_v1/`。

## 项目结构

```
English optimizing/
├── .env                    # 环境变量配置（包含API密钥）
├── .gitignore             # Git忽略文件配置
├── generate_sft_data.py    # 一次性生成SFT训练数据
├── data_gen.py            # 原始数据生成脚本
├── data_expansion.py      # 数据扩展脚本
├── filter.py              # 数据过滤脚本
├── filter_sft.py          # SFT数据过滤脚本
├── fix_seed.py            # 种子数据修复脚本
├── load_model.py          # 模型加载脚本
├── train/                 # 训练相关文件
│   └── train_sft.py       # SFT监督微调脚本
├── models/                # 本地模型存储目录（被Git忽略）
├── sft_model/             # 微调后模型保存目录（被Git忽略）
├── offload_temp/          # 模型offload临时目录（被Git忽略）
├── sft_train.jsonl        # 原始SFT训练数据
├── sft_train_filtered.jsonl  # 过滤后的SFT训练数据（被Git忽略）
├── sft_train_direct.jsonl # 直接生成的SFT训练数据（被Git忽略）
└── evaluate_results.json  # 评估结果（被Git忽略）
```

## 被忽略文件/目录的存储路径提示

以下文件和目录由于包含大型文件或敏感信息，未被Git上传：

### 1. 模型相关目录
- **models/**: 本地模型存储目录
  - 路径：`./models/`
  - 包含从Hugging Face下载的原始模型（如Qwen2.5-3B-Instruct）
  - 结构示例：`./models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/{commit_hash}/`

- **sft_model/**: 微调后模型保存目录
  - 路径：`./sft_model/`
  - 包含训练完成后的LoRA权重和配置文件

- **offload_temp/**: 模型offload临时目录
  - 路径：`./offload_temp/`
  - 用于模型加载过程中的临时文件

### 2. 数据文件
- **sft_train_filtered.jsonl**: 过滤后的SFT训练数据
  - 路径：`./sft_train_filtered.jsonl`
  - 由 `filter_sft.py` 生成

- **sft_train_direct.jsonl**: 直接生成的SFT训练数据
  - 路径：`./sft_train_direct.jsonl`
  - 由 `generate_sft_data.py` 生成

### 3. 配置和结果文件
- **.env**: 环境变量配置文件
  - 路径：`./.env`
  - 包含Kimi API密钥等敏感信息
  - 格式：`KIMI_API_KEY=your_api_key_here`

- **evaluate_results.json**: 评估结果
  - 路径：`./evaluate_results.json`
  - 包含模型评估的详细结果

## 智能体维护指南

### 在新环境中恢复项目

1. **克隆代码仓库**
   ```bash
   git clone <repository_url> English optimizing
   cd English optimizing
   ```

2. **创建环境变量文件**
   - 复制 `.env.example`（如果存在）或创建新的 `.env` 文件
   - 添加Kimi API密钥：`KIMI_API_KEY=your_api_key_here`

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   # 或单独安装
   pip install openai python-dotenv transformers peft torch datasets
   ```

4. **下载模型**
   - 运行 `load_model.py` 自动下载并保存模型到 `./models/` 目录
   ```bash
   python load_model.py
   ```

5. **生成训练数据**
   - 运行 `generate_sft_data.py` 一次性生成2000条SFT训练数据
   ```bash
   python generate_sft_data.py
   ```

6. **过滤训练数据**
   - 运行 `filter_sft.py` 过滤生成的数据
   ```bash
   python filter_sft.py
   ```

7. **开始训练**
   - 运行 `train/train_sft.py` 开始SFT微调
   ```bash
   python train/train_sft.py
   ```

### 日常维护

1. **更新模型**
   - 修改 `load_model.py` 中的模型名称，运行脚本自动下载新模型

2. **扩展训练数据**
   - 运行 `generate_sft_data.py` 生成更多训练数据
   - 或使用 `data_gen.py` 和 `data_expansion.py` 生成和扩展数据

3. **调整训练参数**
   - 修改 `train/train_sft.py` 中的训练参数，如batch size、learning rate等

4. **监控训练过程**
   - 查看训练日志，监控损失值变化
   - 使用TensorBoard查看训练指标

## 故障排查

### 常见问题

1. **API限流错误**
   - 症状：`Error code: 429 - The engine is currently overloaded`
   - 解决方案：脚本已内置请求频率控制，确保API密钥有效且余额充足

2. **模型加载失败**
   - 症状：`ValueError: Unrecognized model in ../models`
   - 解决方案：运行 `load_model.py` 下载完整模型和tokenizer

3. **内存不足**
   - 症状：`CUDA out of memory`
   - 解决方案：调整 `train_sft.py` 中的 `per_device_train_batch_size` 和 `gradient_accumulation_steps`

4. **数据格式错误**
   - 症状：`JSONDecodeError` 或数据解析失败
   - 解决方案：检查数据文件格式，确保为有效的JSONL格式

## 项目贡献

1. **添加新的语法点**
   - 修改 `generate_sft_data.py` 中的 `GRAMMAR_TOPICS` 列表

2. **优化数据生成**
   - 改进 `generate_sft_data.py` 中的prompt设计
   - 增加更多多样化的句子主题

3. **提升模型性能**
   - 调整LoRA配置参数
   - 尝试不同的训练策略

4. **扩展功能**
   - 添加模型评估脚本
   - 实现模型部署功能

---

此文档将帮助您在任何环境中快速恢复和维护项目，确保模型和数据的正确存储和使用。
