#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本：对比加载LoRA和不加载LoRA的模型纠错效果
用途：直观展示LoRA训练对模型语法纠错能力的提升
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re

def load_base_model(base_model_path):
    """加载基础模型"""
    print("🔄 加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ 基础模型加载完成")
    return model, tokenizer

def load_lora_model(base_model, base_model_path, lora_path):
    """在基础模型上加载LoRA"""
    print(f"🔄 加载LoRA模型: {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ LoRA模型加载完成")
    return model, tokenizer

def correct_sentence(model, tokenizer, error_sentence, model_name="模型"):
    """使用模型纠正错误句子"""
    prompt = f"纠正以下句子的语法错误\n{error_sentence}"
    messages = [{"role": "user", "content": prompt}]
    
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    
    # 生成设置保持一致
    output = model.generate(
        input_ids,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.1
    )
    
    # 解码完整响应
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 提取assistant的回复部分
    if "assistant\n" in full_response:
        assistant_response = full_response.split("assistant\n", 1)[1].strip()
    else:
        # 如果没有明确的assistant标记，尝试提取最后一部分
        assistant_response = full_response.split("user\n")[-1].strip() if "user\n" in full_response else full_response
    
    return assistant_response

def extract_correction_details(response):
    """从模型回复中提取正确句子和原始解释"""
    # 由于correct_sentence已经返回了清理后的assistant回复，直接返回即可
    return response

def evaluate_model():
    """评估主函数"""
    print("="*60)
    print("🎯 LoRA模型训练效果评估")
    print("="*60)
    
    # 模型路径配置
    base_model_path = "./models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
    lora_path = "./sft_model"
    
    # 测试用例（包含各种语法错误类型）
    test_cases = [
        "He said he goes to the store yesterday.",    # 间接引语时态
        "The book which you recommended it is great.", # 定语从句冗余
        "She don't like apples.",                     # 主谓一致
        "I have been to the park last week.",         # 完成时错误
        "They is playing football now.",              # 复数主谓一致
        "My sister go to school every day.",          # 第三人称单数
        "We will meeting tomorrow.",                  # 将来时结构
    ]
    
    # 1. 加载基础模型（不加载LoRA）
    base_model, base_tokenizer = load_base_model(base_model_path)
    base_model.eval()  # 设置为推理模式
    
    # 2. 加载LoRA模型
    lora_model, lora_tokenizer = load_lora_model(base_model, base_model_path, lora_path)
    lora_model.eval()  # 设置为推理模式
    
    print("\n" + "="*60)
    print("📊 开始对比评估...")
    print("="*60)
    
    # 3. 对比测试
    results = []
    for i, error_sentence in enumerate(test_cases, 1):
        print(f"\n=== 测试用例 {i} ===")
        print(f"❌ 原始错误句子: {error_sentence}")
        
        # 基础模型输出
        base_response = correct_sentence(base_model, base_tokenizer, error_sentence, "基础模型")
        
        # LoRA模型输出
        lora_response = correct_sentence(lora_model, lora_tokenizer, error_sentence, "LoRA模型")
        
        # 显示结果
        print("\n📋 对比结果:")
        print(f"   🟢 基础模型:")
        print(f"       模型解释: {base_response}")
        print(f"   🔴 LoRA模型:")
        print(f"       模型解释: {lora_response}")
        
        # 保存结果
        results.append({
            "test_case": i,
            "error_sentence": error_sentence,
            "base_full_response": base_response,
            "lora_full_response": lora_response
        })
        
        print("-" * 50)
    
    # 4. 生成评估报告
    print("\n" + "="*60)
    print("📝 评估报告总结")
    print("="*60)
    print(f"测试用例总数: {len(test_cases)}")
    print(f"基础模型路径: {base_model_path}")
    print(f"LoRA模型路径: {lora_path}")
    
    
    # 保存详细结果到文件
    import json
    with open("evaluate_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n✅ 详细结果已保存到: evaluate_results.json")
    print("="*60)

if __name__ == "__main__":
    evaluate_model()