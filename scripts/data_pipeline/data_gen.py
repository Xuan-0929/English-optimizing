# data_gen.py - WSL2最终版
import os
import openai
from dotenv import load_dotenv
import json
import time

# 加载环境变量
load_dotenv()

# Kimi API配置
client = openai.OpenAI(
    api_key=os.getenv("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

# 10个核心语法点
GRAMMAR_TOPICS = {
    "时态一致": "He said he was going to the store.",
    "虚拟语气": "If I had known, I would have told you.",
    "定语从句": "The book that you recommended is great.",
    "非谓语动词": "Having finished his work, he went home.",
    "主谓一致": "Neither of the solutions is correct.",
    "倒装句": "Only then did I understand the problem.",
    "连接词": "Although it was raining, we went out.",
    "冠词": "I saw a cat and the cat was black.",
    "介词": "She is interested in learning English.",
    "词性误用": "His advice was very helpful."
}

def generate_seed_data(topic, correct_example):
    """生成10条同一语法点的数据"""
    prompt = f"""
    你是一名英语语法专家。请针对"{topic}"语法点，生成错误句子和正确改写的配对数据。
    
    要求：
    1. 参考例句："{correct_example}"
    2. 错误句子要真实、自然，符合中国学生常见错误
    3. 解释要符合句子错误原因，不能是通用解释
    5. 错误句子和正确句子必须是英文
    6. 格式严格为JSON：
    {{
      "error_sentence": "...",
      "correct_sentence": "...",
      "error_type": "{topic}",
      "explanation": "中文解释，50字以内"
    }}
    
    生成10条不同的数据，每条用换行符分隔。
    """
    
    try:
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"生成失败 '{topic}': {e}")
        return None

def main():
    all_data = []
    for topic, example in GRAMMAR_TOPICS.items():
        print(f"正在生成 '{topic}' 数据...")
        try:
            raw_data = generate_seed_data(topic, example)
            if raw_data:
                # 记录当前all_data长度
                len_before = len(all_data)
                
                # 移除Markdown代码块标记
                cleaned_data = raw_data.replace('```json', '').replace('```', '').strip()
                
                # 按JSON对象分隔并解析
                json_objects = []
                current_json = []
                brace_count = 0
                
                for line in cleaned_data.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    for char in line:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                    
                    current_json.append(line)
                    
                    if brace_count == 0 and current_json:
                        try:
                            json_str = ' '.join(current_json)
                            data = json.loads(json_str)
                            if "error_sentence" in data and "correct_sentence" in data:
                                json_objects.append(data)
                        except json.JSONDecodeError:
                            pass
                        current_json = []
                
                # 将解析出的JSON对象添加到all_data
                all_data.extend(json_objects)
                generated_count = len(json_objects)
                print(f"  ✓ 成功生成 {generated_count} 条")
                time.sleep(2)  # 避免API限速
            else:
                print(f"  ✗ 未获取到数据")
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
    
    # 保存所有数据
    with open('seed_data.jsonl', 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 生成完成！共{len(all_data)}条数据")
    print(f"数据已保存到 seed_data.jsonl (JSONL格式)")
    if all_data:
        print(f"示例数据: {all_data[0]}")

if __name__ == "__main__":
    main()
