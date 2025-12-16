# 环境: pip install openai gradio
# 使用方法：
# 1. 安装依赖包: pip install openai gradio
# 2. 运行程序后在浏览器中打开Gradio界面进行翻译练习

import os
import gradio as gr
from openai import OpenAI

def generate_chinese_sentence(level="托福100分", length="中等长度"):
    """生成一个中文句子供用户翻译"""
    try:
        client = OpenAI(
            api_key = "sk-HFY7OtHfcseybS5da7CkVr9xt8avygpKBtHHeNzex2kSJgvC",
            base_url = "https://api.moonshot.cn/v1",
        )
        
        # 根据用户选择的长度生成对应的描述
        length_desc = {
            "短句子": "10-15字左右",
            "中等长度": "20-30字左右",
            "长句子": "30字以上"
        }[length]
        
        response = client.chat.completions.create(
            model = "kimi-k2-turbo-preview",
            messages = [
                {"role": "system", "content": f"你是一个英语教学助手。请生成一个{level}难度的中文句子，适合该水平的学生翻译练习。句子长度为{length_desc}，应当表达一个完整的意思，主题可以是日常生活、学习、工作等常见场景。只输出中文句子，不要有其他解释或说明。"},
                {"role": "user", "content": f"我是一名{level}水平的学生，请生成一个{length}的中文句子用于翻译练习。"}
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
            api_key = "sk-HFY7OtHfcseybS5da7CkVr9xt8avygpKBtHHeNzex2kSJgvC",
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
    
    # 提供可选的水平和长度选项
    levels = ["托福80分", "托福100分", "托福110分", "托福120分"]
    lengths = ["短句子", "中等长度", "长句子"]
    
    while True:
        # 获取用户水平选择
        print("请选择你的英语水平：")
        for i, level in enumerate(levels, 1):
            print(f"{i}. {level}")
        level_choice = input(f"请输入1-{len(levels)}之间的数字: ")
        try:
            level = levels[int(level_choice) - 1]
        except:
            print("无效输入，使用默认水平：托福100分")
            level = "托福100分"
        
        # 获取用户长度选择
        print(f"\n请选择句子长度：")
        for i, length in enumerate(lengths, 1):
            print(f"{i}. {length}")
        length_choice = input(f"请输入1-{len(lengths)}之间的数字: ")
        try:
            length = lengths[int(length_choice) - 1]
        except:
            print("无效输入，使用默认长度：中等长度")
            length = "中等长度"
        
        # 生成中文句子
        print(f"\n正在生成{level}难度、{length}的练习句子...")
        chinese_sentence = generate_chinese_sentence(level, length)
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

# Gradio界面相关函数
def translate_and_evaluate(chinese_sentence, user_translation):
    """处理翻译和评估的Gradio接口函数"""
    if not user_translation.strip():
        return "请输入你的英文翻译"
    return evaluate_translation(chinese_sentence, user_translation)

def get_new_sentence(level, length):
    """生成新的中文句子"""
    return generate_chinese_sentence(level, length)

# 创建Gradio界面
with gr.Blocks(title="英语翻译练习系统") as demo:
    gr.Markdown("# 英语翻译练习系统")
    
    # 添加用户需求模块
    with gr.Row():
        with gr.Column(scale=1):
            level = gr.Dropdown(
                choices=["托福80分", "托福100分", "托福110分", "托福120分"],
                label="英语水平",
                value="托福100分"
            )
        with gr.Column(scale=1):
            length = gr.Dropdown(
                choices=["短句子", "中等长度", "长句子"],
                label="句子长度",
                value="中等长度"
            )
    
    with gr.Row():
        with gr.Column():
            chinese_sentence = gr.Textbox(label="中文句子", lines=3, interactive=False)
            new_sentence_btn = gr.Button("生成新句子")
        
        with gr.Column():
            user_translation = gr.Textbox(label="你的英文翻译", lines=3, placeholder="请输入你的英文翻译")
            evaluate_btn = gr.Button("评估翻译")
    
    feedback = gr.Textbox(label="翻译评估结果", lines=10, interactive=False)
    
    # 设置事件处理
    new_sentence_btn.click(fn=get_new_sentence, inputs=[level, length], outputs=chinese_sentence)
    evaluate_btn.click(fn=translate_and_evaluate, inputs=[chinese_sentence, user_translation], outputs=feedback)
    
    # 初始化时生成第一个句子
    demo.load(fn=get_new_sentence, inputs=[level, length], outputs=chinese_sentence)

if __name__ == "__main__":
    # 启动Gradio界面
    demo.launch()
    # 如果需要使用命令行版本，可以取消下面的注释
    # translation_practice()