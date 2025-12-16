# 环境: pip install openai
# 使用方法：
# 1. 安装openai包: pip install openai
# 2. 运行程序后按照提示进行交互式翻译练习

import os
import gradio as gr
from openai import OpenAI

def generate_chinese_sentence():
    """生成一个中文句子供用户翻译"""
    try:
        client = OpenAI(
            api_key = "sk-oY4wLWPvZJcEyBznpjNHJlnOKy8ulP8cExa7Kg8KHTLB2odD",
            base_url = "https://api.moonshot.cn/v1",
        )
        
        response = client.chat.completions.create(
            model = "kimi-k2-turbo-preview",
            messages = [
                {"role": "system", "content": "你是一个英语教学助手。请生成一个托福100分左右难度的中文句子，适合备考托福的学生翻译练习。句子应当表达一个完整的意思，主题可以是日常生活、学习、工作等常见场景。只输出中文句子，不要有其他解释或说明。"},
                {"role": "user", "content": "我是一名备考托福的学生，请生成一个适合英语学习者翻译的中文句子。"}
            ],
            temperature = 0.7,
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"生成中文句子时出错: {str(e)}")
        return "今天天气真好，我们一起去公园散步吧。"

def evaluate_translation(chinese_sentence, user_translation):
    """评估用户的翻译并提供反馈"""
    try:
        client = OpenAI(
            api_key = "sk-oY4wLWPvZJcEyBznpjNHJlnOKy8ulP8cExa7Kg8KHTLB2odD",
            base_url = "https://api.moonshot.cn/v1",
        )
        
        prompt = f"""
        原始中文句子: {chinese_sentence}
        用户的英文翻译: {user_translation}
        
        请作为一名英语教师，评估这个翻译并提供以下信息：
        1. 准确性评价：翻译是否准确传达了原句的意思
        2. 语法检查：翻译中是否存在语法错误
        3. 地道性评价：翻译是否自然地道
        4. 改进建议：如何改进这个翻译
        5. 参考翻译：提供一个更准确、地道的翻译版本
        6. 评分：从0到10分评价这个翻译的质量
        
        请以友好、鼓励的语气用中文回答，并提供具体的改进建议。
        """
        
        response = client.chat.completions.create(
            model = "kimi-k2-turbo-preview",
            messages = [
                {"role": "system", "content": "你是一位经验丰富的英语教师，善于评估学生的翻译并提供建设性的反馈。"},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.6,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"评估翻译时出错: {str(e)}")
        return "无法评估翻译，请稍后再试。"

def translation_practice():
    """交互式翻译练习主函数"""
    print("====== 英语翻译练习系统 ======\n")
    
    while True:
        # 生成中文句子
        print("正在生成练习句子...")
        chinese_sentence = generate_chinese_sentence()
        print(f"\n请将下面的中文句子翻译成英文:\n{chinese_sentence}\n")
        
        # 获取用户翻译
        user_translation = input("请输入你的英文翻译: ")
        
        # 评估翻译
        print("\n正在评估你的翻译...\n")
        feedback = evaluate_translation(chinese_sentence, user_translation)
        print("===== 翻译评估结果 =====")
        print(feedback)
        print("====================\n")
        
        # 询问是否继续
        again = input("是否继续练习？(y/n): ")
        if again.lower() != 'y':
            print("\n谢谢使用！再见！")
            break


if __name__ == "__main__":
    translation_practice()