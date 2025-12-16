# ai_filter.py - AI质量评估
import os
import json
import openai
from dotenv import load_dotenv
import time

load_dotenv()
client = openai.OpenAI(
    api_key=os.getenv("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

def evaluate_batch(items):
    """批量评估10条数据"""
    prompt = f"""
    请批量评估以下英语语法纠错数据的质量，格式为{{"error_sentence": "英文错误句子", "correct_sentence": "英文正确句子", "error_type": "错误类型", "explanation": "中文解释"}}
    每项从6个维度打分（1-5分）：
    
    评分标准：
    1. **错误真实性**：是否真实反映中国学生常见错误？（1=刻意编造，5=非常自然）
    2. **改正正确性**：改正后句子是否完全正确？（1=仍有错误，5=完美）
    3. **类型准确性**：error_type标注是否准确？（1=错误，5=精确）
    4. **解释清晰度**：explanation是否正确、与错误句子相关？（1=错误，5=正确）
    5. **格式正确**：error_sentence和correct_sentence中是否有中文？（1=有中文，5=无中文）
    6. **脱裤子放屁**：error_sentence与correct_sentence相比是否有更改或提升的必要？（1=否，5=是）

    要求：
    - 总分≥25分且单项≥4分为合格
    - 只输出合格项的索引（从0开始）
    - 格式：`合格索引: 0,3,5,9`
    
    待评估数据：
    {json.dumps(items, ensure_ascii=False, indent=2)}
    """
    
    try:
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"评估失败: {e}")
        return None

def main():
    # 读取原始数据
    with open('seed_data.jsonl', 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"开始评估 {len(raw_data)} 条数据...")
    
    # 批量处理（每批10条）
    batch_size = 10
    qualified_indices = []
    
    for i in range(0, len(raw_data), batch_size):
        batch = raw_data[i:i+batch_size]
        print(f"评估第 {i+1}-{min(i+batch_size, len(raw_data))} 条...")
        
        result = evaluate_batch(batch)
        if result and "合格索引:" in result:
            # 解析合格索引
            try:
                indices_str = result.split("合格索引:")[1].strip()
                if indices_str:  # 确保不为空
                    batch_qualified = [int(x.strip()) for x in indices_str.split(",") if x.strip()]
                    # 转换为全局索引
                    qualified_indices.extend([i + idx for idx in batch_qualified])
            except Exception as e:
                print(f"  解析索引失败: {result}, 错误: {e}")
        
        time.sleep(1)  # 避免API限速
    
    # 筛选合格数据
    filtered = [raw_data[idx] for idx in sorted(qualified_indices)]
    
    print(f"\n筛选结果: {len(raw_data)} → {len(filtered)} 条")
    print(f"保留率: {len(filtered)/len(raw_data)*100:.1f}%")
    
    # 保存
    with open('seed_data_filtered.jsonl', 'w', encoding='utf-8') as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("✅ 已保存到 seed_data_filtered.jsonl")
    
if __name__ == "__main__":
    main()
