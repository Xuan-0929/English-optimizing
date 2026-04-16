# generate_sft_data.py - 一次性生成2000条高质量语法纠错数据
import os
import openai
from dotenv import load_dotenv
import json
import time
import concurrent.futures
from collections import defaultdict

# 加载环境变量
load_dotenv()

# Kimi API配置
client = openai.OpenAI(
    api_key=os.getenv("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

# 10个核心语法点
GRAMMAR_TOPICS = [
    "时态一致",
    "虚拟语气",
    "定语从句",
    "非谓语动词",
    "主谓一致",
    "倒装句",
    "连接词",
    "冠词",
    "介词",
    "词性误用"
]

# 每个语法点生成的示例数
EXAMPLES_PER_TOPIC = 200

# 总目标数据量
TARGET_TOTAL = 2000

def generate_grammar_data(topic, examples_count):
    """为单个语法点生成指定数量的语法纠错数据"""
    prompt = f"""
    你是一名专业的英语语法教师，擅长设计适合中国学生的语法纠错练习。
    请为以下语法点生成 {examples_count} 条高质量的语法纠错数据。
    
    语法点：{topic}
    
    生成要求：
    1. 每条数据必须包含：error_sentence（错误句子）、correct_sentence（正确句子）、error_type（错误类型）、explanation（中文解释）
    2. 错误句子必须是真实的、符合中国学生常见错误
    3. 错误类型必须明确为指定的语法点
    4. 解释必须具体、准确，针对该错误，50字以内
    5. 句子主题要多样化，包括日常生活、学习、工作等场景
    6. 避免生成相似或重复的句子
    7. 所有句子必须是英文
    8. 严格使用JSON格式，每条数据占一行，不要使用markdown代码块
    9. 不要在输出中添加任何额外的解释或说明
    
    输出格式（每行一条）：
    {{"error_sentence":"错误句子","correct_sentence":"正确句子","error_type":"{topic}","explanation":"中文解释"}}
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=8000  # 增加token限制，确保一次性生成足够数据
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"生成 {topic} 失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
    return None

def parse_generated_data(raw_data, topic):
    """解析生成的数据，验证格式并去重"""
    parsed_data = []
    duplicate_checker = set()
    
    # 按行处理
    lines = raw_data.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        try:
            # 解析JSON
            data = json.loads(line)
            
            # 验证必要字段
            required_fields = ["error_sentence", "correct_sentence", "error_type", "explanation"]
            if not all(field in data for field in required_fields):
                continue
            
            # 验证error_type是否正确
            if data["error_type"] != topic:
                continue
            
            # 验证句子长度和质量
            if len(data["error_sentence"]) < 5 or len(data["correct_sentence"]) < 5:
                continue
            
            # 去重检查
            unique_key = data["error_sentence"].strip().lower()
            if unique_key not in duplicate_checker:
                duplicate_checker.add(unique_key)
                parsed_data.append(data)
                
                # 如果达到目标数量，提前结束
                if len(parsed_data) >= EXAMPLES_PER_TOPIC:
                    break
                    
        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(f"解析数据失败: {e}")
            continue
    
    return parsed_data

def convert_to_sft_format(data):
    """将数据转换为SFT训练格式"""
    sft_data = []
    for item in data:
        sft_record = {
            "instruction": "纠正以下句子的语法错误",
            "input": item["error_sentence"],
            "output": f"**错误类型**: {item['error_type']}\n**改正**: {item['correct_sentence']}\n**解释**: {item['explanation']}"
        }
        sft_data.append(sft_record)
    return sft_data

def main():
    print(f"开始一次性生成 {TARGET_TOTAL} 条语法纠错SFT数据...")
    print(f"10个语法点，每个语法点生成 {EXAMPLES_PER_TOPIC} 条\n")
    
    all_data = []
    
    # 使用串行生成，降低API负载
    for topic in GRAMMAR_TOPICS:
        print(f"\n🔄 正在生成 {topic}...")
        raw_data = generate_grammar_data(topic, EXAMPLES_PER_TOPIC)
        if raw_data:
            parsed = parse_generated_data(raw_data, topic)
            all_data.extend(parsed)
            print(f"✅ {topic}: 生成 {len(parsed)} 条数据")
        else:
            print(f"❌ {topic}: 生成失败")
        
        # 控制请求频率，避免API限流
        print("⏱️  等待1秒，控制API请求频率...")
        time.sleep(1)
    
    # 最终数据统计
    print(f"\n📊 生成结果统计:")
    print(f"总生成数据: {len(all_data)} 条")
    
    # 按语法点统计
    topic_counts = defaultdict(int)
    for item in all_data:
        topic_counts[item["error_type"]] += 1
    
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count} 条")
    
    # 去重最终检查
    final_checker = set()
    final_data = []
    for item in all_data:
        key = item["error_sentence"].strip().lower()
        if key not in final_checker:
            final_checker.add(key)
            final_data.append(item)
    
    print(f"\n🔍 最终去重后: {len(final_data)} 条")
    
    # 如果数据不足，补充生成
    if len(final_data) < TARGET_TOTAL:
        print(f"\n⚠️ 数据不足 {TARGET_TOTAL} 条，正在补充生成...")
        remaining = TARGET_TOTAL - len(final_data)
        
        # 为所有语法点平均补充
        supplement_per_topic = remaining // len(GRAMMAR_TOPICS) + 1
        
        for topic in GRAMMAR_TOPICS:
            if len(final_data) >= TARGET_TOTAL:
                break
                
            raw_data = generate_grammar_data(topic, supplement_per_topic)
            if raw_data:
                parsed = parse_generated_data(raw_data, topic)
                # 去重并添加
                for item in parsed:
                    if len(final_data) >= TARGET_TOTAL:
                        break
                    key = item["error_sentence"].strip().lower()
                    if key not in final_checker:
                        final_checker.add(key)
                        final_data.append(item)
        
        print(f"✅ 补充生成后: {len(final_data)} 条")
    
    # 转换为SFT格式
    sft_data = convert_to_sft_format(final_data)
    
    # 保存数据
    with open('sft_train_direct.jsonl', 'w', encoding='utf-8') as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n🎉 生成完成！")
    print(f"📁 数据已保存到: sft_train_direct.jsonl")
    print(f"📊 最终生成: {len(sft_data)} 条SFT训练数据")

if __name__ == "__main__":
    main()