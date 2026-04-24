# 本地模型测试脚本 - 语法纠错效果测试
# 环境: pip install torch transformers peft gradio
# 使用方法：
# 1. 确保已训练好模型（sft_model目录存在）
# 2. 运行程序后按照提示进行交互式语法纠错练习

import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr

try:
    from scripts.evaluation.type_constraints import apply_type_constraints
except ModuleNotFoundError:
    apply_type_constraints = None

# 配置参数
MODEL_PATH = "./models/Qwen2.5-3B-Instruct"  # 基础模型路径
LORA_PATH = "./train/sft_model"  # LoRA模型路径
ERROR_TYPE_RE = re.compile(r"(\*\*错误类型\*\*:\s*)(.+?)(\n)")
CORRECTION_RE = re.compile(r"\*\*改正\*\*:\s*(.+?)\n", re.S)

def load_local_model():
    """加载本地训练的模型"""
    print("正在加载本地模型...")
    
    # 4-bit量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # 加载基础模型
    print(f"加载基础模型: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        dtype=torch.float16,
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载LoRA适配器
    if os.path.exists(LORA_PATH):
        print(f"加载LoRA适配器: {LORA_PATH}")
        model = PeftModel.from_pretrained(model, LORA_PATH)
        print("✅ LoRA模型加载成功")
    else:
        print(f"⚠️  LoRA路径不存在: {LORA_PATH}")
        print("将使用基础模型进行测试")
    
    model.eval()
    return model, tokenizer

def generate_correction(model, tokenizer, user_sentence):
    """使用模型生成语法纠错"""
    # 构造prompt
    prompt = f"纠正以下句子的语法错误\n{user_sentence}\n"
    
    # 应用chat模板
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if apply_type_constraints is not None:
        m_corr = CORRECTION_RE.search(generated_text)
        m_type = ERROR_TYPE_RE.search(generated_text)
        if m_corr and m_type:
            correction = m_corr.group(1).strip()
            pred_type = m_type.group(2).strip()
            final_type, _ = apply_type_constraints(user_sentence, correction, pred_type)
            if final_type != pred_type:
                generated_text = ERROR_TYPE_RE.sub(rf"\1{final_type}\3", generated_text, count=1)
    return generated_text

def evaluate_correction(user_sentence, model_output):
    """评估纠错结果"""
    evaluation = {
        "user_sentence": user_sentence,
        "model_output": model_output,
        "has_correction": "**改正**" in model_output,
        "has_error_type": "**错误类型**" in model_output,
        "has_explanation": "**解释**" in model_output,
    }
    return evaluation

def test_single_sentence(model, tokenizer, sentence):
    """测试单个句子"""
    print(f"\n{'='*50}")
    print(f"测试句子: {sentence}")
    print(f"{'='*50}")
    
    result = generate_correction(model, tokenizer, sentence)
    print(f"\n模型输出:\n{result}")
    
    evaluation = evaluate_correction(sentence, result)
    print(f"\n评估结果:")
    print(f"包含改正: {evaluation['has_correction']}")
    print(f"包含错误类型: {evaluation['has_error_type']}")
    print(f"包含解释: {evaluation['has_explanation']}")
    
    return evaluation
def batch_test(model, tokenizer, test_sentences):
    """批量测试"""
    print(f"\n{'='*60}")
    print("批量测试模式")
    print(f"{'='*60}\n")
    
    results = []
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n[{i}/{len(test_sentences)}] 测试中...")
        result = test_single_sentence(model, tokenizer, sentence)
        results.append(result)
    
    # 统计结果
    print(f"\n{'='*60}")
    print("测试统计")
    print(f"{'='*60}")
    
    total = len(results)
    has_correction = sum(1 for r in results if r['has_correction'])
    has_error_type = sum(1 for r in results if r['has_error_type'])
    has_explanation = sum(1 for r in results if r['has_explanation'])
    
    print(f"总测试数: {total}")
    print(f"包含改正: {has_correction}/{total} ({has_correction/total*100:.1f}%)")
    print(f"包含错误类型: {has_error_type}/{total} ({has_error_type/total*100:.1f}%)")
    print(f"包含解释: {has_explanation}/{total} ({has_explanation/total*100:.1f}%)")
    
    return results

def interactive_test(model, tokenizer):
    """交互式测试"""
    print("\n" + "="*60)
    print("交互式语法纠错测试")
    print("="*60)
    print("输入包含语法错误的英文句子，模型将进行纠错")
    print("输入 'quit' 退出程序")
    print("输入 'batch' 进入批量测试模式")
    
    while True:
        user_input = input("\n请输入测试句子: ").strip()
        
        if user_input.lower() == 'quit':
            print("\n再见！")
            break
        elif user_input.lower() == 'batch':
            # 批量测试
            test_sentences = [
                "I am going to the store and buy some apples.",
                "She said she go to the party last night.",
                "He told me that he is visiting his parents this weekend.",
                "I have been studying for two hours and finish my homework.",
                "They always was on time for work.",
                "I wish I can go to the party tonight.",
                "If I were you, I will study hard.",
                "The movie who I watched last night was very boring.",
                "She likes swim in the lake.",
                "I usually go to park after work.",
            ]
            batch_test(model, tokenizer, test_sentences)
        elif user_input:
            test_single_sentence(model, tokenizer, user_input)

def create_gradio_interface(model, tokenizer):
    """创建Gradio界面"""
    def correct_grammar(sentence):
        if not sentence.strip():
            return "请输入一个句子。"
        
        result = generate_correction(model, tokenizer, sentence)
        return result
    
    with gr.Blocks(title="英语语法纠错系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 英语语法纠错系统
        
        输入包含语法错误的英文句子，系统将自动纠正并提供解释。
        
        ## 使用说明
        1. 在下方输入框中输入英文句子
        2. 点击"纠正语法"按钮
        3. 查看纠错结果和解释
        """)
        
        with gr.Row():
            with gr.Column():
                input_box = gr.Textbox(
                    label="输入句子",
                    placeholder="请输入包含语法错误的英文句子...",
                    lines=3
                )
                submit_btn = gr.Button("纠正语法", variant="primary")
            
            with gr.Column():
                output_box = gr.Textbox(
                    label="纠错结果",
                    lines=8,
                    interactive=False
                )
        
        submit_btn.click(
            fn=correct_grammar,
            inputs=input_box,
            outputs=output_box
        )
        
        gr.Examples(
            examples=[
                "I am going to the store and buy some apples.",
                "She said she go to the party last night.",
                "He told me that he is visiting his parents this weekend.",
                "I have been studying for two hours and finish my homework.",
                "I wish I can go to the party tonight.",
                "If I were you, I will study hard.",
                "The movie who I watched last night was very boring.",
                "She likes swim in the lake.",
                "I usually go to park after work.",
            ],
            inputs=input_box,
            label="示例句子"
        )
    
    return demo

def main():
    """主函数"""
    print("\n" + "="*60)
    print("本地模型测试系统")
    print("="*60)
    
    # 加载模型
    model, tokenizer = load_local_model()
    
    # 选择模式
    print("\n请选择测试模式:")
    print("1. 交互式命令行测试")
    print("2. Gradio网页界面测试")
    print("3. 批量测试")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == '1':
        interactive_test(model, tokenizer)
    elif choice == '2':
        print("\n启动Gradio界面...")
        demo = create_gradio_interface(model, tokenizer)
        demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
    elif choice == '3':
        test_sentences = [
            "I am going to the store and buy some apples.",
            "She said she go to the party last night.",
            "He told me that he is visiting his parents this weekend.",
            "I have been studying for two hours and finish my homework.",
            "They always was on time for work.",
            "He used to get up early, but now he is sleeping.",
            "I will be going to the store and buy some milk.",
            "She said she has been studying English for five years.",
            "I was walking in the park when it rained.",
            "When I arrived at the station, the train leaves.",
        ]
        batch_test(model, tokenizer, test_sentences)
    else:
        print("无效选择，默认使用交互式模式")
        interactive_test(model, tokenizer)

if __name__ == "__main__":
    main()
