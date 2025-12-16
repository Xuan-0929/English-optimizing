# train_sft.py - 修复版（硬编码路径 + 强制冻结LoRA）
import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import json
from datetime import datetime

def load_sft_data(data_path):
    """加载SFT格式数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]
    
    # 转换为ChatML格式
    formatted_data = []
    for item in raw_data:
        messages = [
            {"role": "user", "content": f"{item['instruction']}\n{item['input']}"},
            {"role": "assistant", "content": item['output']}
        ]
        formatted_data.append({"messages": messages})
    
    return formatted_data

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize对话数据"""
    texts = []
    for example in examples['messages']:
        text = tokenizer.apply_chat_template(example, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

def main():
    # ========== 核心配置（修改这里） ==========
    MODEL_PATH = "./models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"  # ✅ 使用相对路径
    DATA_PATH = "./sft_train_filtered.jsonl"      # ✅ 你的数据文件
    OUTPUT_DIR = "./sft_model"                   # ✅ 输出目录
    
    # 增量训练配置
    INCREMENTAL_TRAINING = False  # ✅ 设置为True表示在现有LoRA基础上继续训练
    EXISTING_LORA_PATH = "./sft_model"  # ✅ 现有LoRA路径
    
    # 操作步骤：
    # 1. 将 models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/xxxxx/ 里的所有文件
    # 2. 复制到 models/Qwen2.5-3B-Instruct/ 目录下
    
    print(f"🚀 开始SFT训练 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 数据文件: {DATA_PATH}")
    print(f"💾 输出目录: {OUTPUT_DIR}")
    print(f"🔄 增量训练: {'是' if INCREMENTAL_TRAINING else '否'}")
    
    # 1. 4-bit量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # 2. 加载模型（保证本地路径）
    print("\n正在加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        dtype=torch.float16,
        local_files_only=True  # ✅ 强制只读本地，不触发网络
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 加载或创建LoRA模型
    if INCREMENTAL_TRAINING:
        # 增量训练：加载现有LoRA
        print(f"\n🔄 加载现有LoRA模型: {EXISTING_LORA_PATH}")
        if not os.path.exists(EXISTING_LORA_PATH):
            raise ValueError(f"❌ 现有LoRA路径不存在: {EXISTING_LORA_PATH}")
        model = PeftModel.from_pretrained(model, EXISTING_LORA_PATH)
        model.print_trainable_parameters()
    else:
        # 常规训练：冻结基础模型参数并创建新LoRA
        print("\n正在冻结基础模型参数...")
        for param in model.parameters():
            param.requires_grad = False
        
        # LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # 必须看到 trainable params: 3,932,160
    
    # 6. 加载数据
    print("\n正在加载数据...")
    train_data = load_sft_data(DATA_PATH)
    print(f"✅ 加载 {len(train_data)} 条训练数据")
    
    # 7. Tokenize
    print("\n正在Tokenize...")
    from datasets import Dataset
    dataset = Dataset.from_list(train_data)
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 8. 训练参数配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        fp16=True,
        optim="adamw_torch",
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        # 重要：禁用评估，防止生成时网络请求
        eval_strategy="no",
        do_eval=False,
        # 重要：确保训练时model.eval()不会被调用
        prediction_loss_only=True,
    )
    
    # 9. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100
        ),
    )
    
    # 10. 开始训练
    print("\n🚀 训练开始...")
    trainer.train()
    
    # 11. 保存
    print("\n💾 保存模型...")
    model.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✅ 训练完成！模型保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
