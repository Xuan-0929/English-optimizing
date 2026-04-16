# expand_data_v2.py - 增强版，确保生成2000条
import os
import json
import openai
from dotenv import load_dotenv
import time
import re
from collections import defaultdict

load_dotenv()
client = openai.OpenAI(
    api_key=os.getenv("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

def generate_variants(seed_data, target_count=40):  # 增加到40条
    """基于种子数据生成变体"""
    prompt = f"""
    基于以下种子数据，生成 {target_count} 个变体。每条必须独立成行，严格JSON格式。
    
    要求：
    1. 保持 error_type="{seed_data['error_type']}"
    2. 保持错误模式：{seed_data['explanation'][:40]}
    3. 替换：主语、宾语、时间、地点、具体名词
    4. 确保 error_sentence 有真实语法错误
    5. 确保 correct_sentence 完全正确
    
    种子数据：
    {json.dumps(seed_data, ensure_ascii=False)}
    
    生成格式（每行一条，不要markdown代码块）：
    {{"error_sentence":"...","correct_sentence":"...","error_type":"...","explanation":"..."}}
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=3000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  第{attempt+1}次尝试失败: {e}")
            time.sleep(2)
    
    return None

def parse_variants(raw_text, seed_type):
    """健壮解析JSON变体"""
    variants = []
    # 移除可能的markdown标记
    cleaned = raw_text.replace('```json', '').replace('```', '').strip()
    
    # 查找所有JSON对象（用正则匹配）
    json_pattern = r'\{[^}]+\}'
    matches = re.finditer(json_pattern, cleaned, re.DOTALL)
    
    for match in matches:
        try:
            variant = json.loads(match.group())
            # 验证字段完整性
            required_keys = {'error_sentence', 'correct_sentence', 'error_type', 'explanation'}
            if required_keys.issubset(variant.keys()):
                variants.append(variant)
        except:
            continue
    
    return variants

def main():
    # 读取种子数据
    with open('seed_data_fixed.jsonl', 'r', encoding='utf-8') as f:
        seeds = [json.loads(line) for line in f]
    
    print(f"开始扩展 {len(seeds)} 条种子 → 目标2000条...")
    
    all_variants = []
    duplicate_checker = defaultdict(set)
    total_needed = 2000
    variants_per_seed = total_needed // len(seeds) + 5  # 每条生成约30条
    
    for idx, seed in enumerate(seeds):
        print(f"\n处理第 {idx+1}/{len(seeds)} 条种子...")
        
        raw_variants = generate_variants(seed, target_count=variants_per_seed)
        
        if raw_variants:
            parsed = parse_variants(raw_variants, seed['error_type'])
            print(f"  原始生成: {len(parsed)} 条")
            
            # 去重并添加
            added = 0
            for v in parsed:
                key = v['error_sentence'].strip().lower()
                if key not in duplicate_checker[v['error_type']]:
                    duplicate_checker[v['error_type']].add(key)
                    all_variants.append(v)
                    added += 1
            
            print(f"  去重后添加: {added} 条")
            print(f"  当前总计: {len(all_variants)} 条")
        
        time.sleep(0.5)  # 更短延迟
    
    # 如果不足2000条，补充生成
    if len(all_variants) < total_needed:
        print(f"\n⚠️ 数量不足 ({len(all_variants)} < {total_needed})，补充生成中...")
        
        # 重新用更高的temperature生成
        for seed in seeds[:10]:  # 取前10条种子再生成一轮
            raw_variants = generate_variants(seed, target_count=50)
            if raw_variants:
                parsed = parse_variants(raw_variants, seed['error_type'])
                for v in parsed:
                    key = v['error_sentence'].strip().lower()
                    if len(all_variants) >= total_needed:
                        break
                    if key not in duplicate_checker[v['error_type']]:
                        duplicate_checker[v['error_type']].add(key)
                        all_variants.append(v)
    
    print(f"\n✅ 最终生成: {len(all_variants)} 条")
    
    # 保存为SFT格式
    with open('sft_train.jsonl', 'w', encoding='utf-8') as f:
        for item in all_variants:
            sft_record = {
                "instruction": "纠正以下句子的语法错误",
                "input": item["error_sentence"],
                "output": f"**错误类型**: {item['error_type']}\n**改正**: {item['correct_sentence']}\n**解释**: {item['explanation']}"
            }
            f.write(json.dumps(sft_record, ensure_ascii=False) + '\n')
    
    print("✅ 已保存为 sft_train.json")

if __name__ == "__main__":
    main()
