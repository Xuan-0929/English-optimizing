import json
import re
from collections import Counter, defaultdict
import os

def analyze_dataset(file_path):
    """分析数据集质量"""
    print(f"正在分析数据集: {file_path}")
    
    # 初始化统计数据
    total_entries = 0
    error_type_counter = Counter()
    explanation_counter = Counter()
    duplicate_inputs = defaultdict(int)
    correct_sentences_modified = 0
    error_explanation_mismatch = 0
    
    # 读取并分析数据
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                entry = json.loads(line)
                total_entries += 1
                
                # 检查输入是否重复
                input_text = entry.get('input', '')
                duplicate_inputs[input_text] += 1
                
                # 解析输出内容
                output = entry.get('output', '')
                
                # 提取错误类型
                error_type_match = re.search(r'\*\*错误类型\*\*:\s*(.*?)\n', output)
                if error_type_match:
                    error_type = error_type_match.group(1).strip()
                    error_type_counter[error_type] += 1
                else:
                    error_type = '未识别'
                    error_type_counter[error_type] += 1
                
                # 提取改正后的句子
                correction_match = re.search(r'\*\*改正\*\*:\s*(.*?)\n', output)
                if correction_match:
                    corrected = correction_match.group(1).strip()
                else:
                    corrected = ''
                
                # 提取解释
                explanation_match = re.search(r'\*\*解释\*\*:\s*(.*?)$', output, re.DOTALL)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                    explanation_counter[explanation] += 1
                else:
                    explanation = ''
                
                # 检查是否修改了正确的句子
                if input_text == corrected:
                    correct_sentences_modified += 1
                
                # 简单检查错误类型与解释是否匹配（示例规则）
                if error_type and explanation:
                    if '时态' in error_type and '时态' not in explanation:
                        error_explanation_mismatch += 1
                    elif '虚拟语气' in error_type and '虚拟' not in explanation:
                        error_explanation_mismatch += 1
                        
            except json.JSONDecodeError:
                print(f"第 {line_num} 行 JSON 格式错误: {line}")
                continue
            except Exception as e:
                print(f"处理第 {line_num} 行时出错: {str(e)}")
                continue
    
    # 输出分析结果
    print(f"\n=== 数据集质量分析报告 ===")
    print(f"总样本数: {total_entries}")
    
    print(f"\n1. 错误类型分布:")
    for error_type, count in error_type_counter.most_common():
        percentage = (count / total_entries) * 100
        print(f"   {error_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n2. 解释重复性分析:")
    total_unique_explanations = len(explanation_counter)
    print(f"   总独特解释数: {total_unique_explanations}")
    print(f"   平均每个解释使用次数: {total_entries / total_unique_explanations:.2f}")
    print(f"   最常见的10个解释:")
    for explanation, count in explanation_counter.most_common(10):
        print(f"   - {explanation}: {count}次")
    
    print(f"\n3. 数据冗余分析:")
    duplicate_count = sum(1 for count in duplicate_inputs.values() if count > 1)
    print(f"   重复输入数: {duplicate_count}")
    print(f"   无错误修改数（输入与输出相同）: {correct_sentences_modified}")
    
    print(f"\n4. 内容一致性分析:")
    print(f"   错误类型与解释不匹配数: {error_explanation_mismatch}")
    
    print(f"\n5. 数据质量评分:")
    # 简单评分机制
    quality_score = 100
    
    # 错误类型多样性扣分（如果单一类型占比超过50%）
    if error_type_counter.most_common(1)[0][1] / total_entries > 0.5:
        quality_score -= 20
    
    # 解释重复性扣分
    if total_unique_explanations < total_entries * 0.3:
        quality_score -= 20
    
    # 重复内容扣分
    if duplicate_count > total_entries * 0.1:
        quality_score -= 15
    
    # 无效修改扣分
    if correct_sentences_modified > total_entries * 0.1:
        quality_score -= 15
    
    # 一致性问题扣分
    if error_explanation_mismatch > total_entries * 0.1:
        quality_score -= 10
    
    print(f"   综合质量评分: {quality_score}/100")
    
    print(f"\n=== 分析完成 ===")
    
    return {
        'total_entries': total_entries,
        'error_types': dict(error_type_counter),
        'explanations': dict(explanation_counter),
        'duplicate_inputs': dict(duplicate_inputs),
        'correct_sentences_modified': correct_sentences_modified,
        'error_explanation_mismatch': error_explanation_mismatch,
        'quality_score': quality_score
    }

if __name__ == "__main__":
    # 默认分析当前实际训练集；其余文件若存在也一并分析
    train_file = "data/sft_train.jsonl"
    original_file = "sft_train_direct.jsonl"
    filtered_file = "data/sft_train_filtered.jsonl"

    if os.path.exists(train_file):
        print("\n" + "=" * 50)
        print("训练集 data/sft_train.jsonl:")
        print("=" * 50)
        analyze_dataset(train_file)

    if os.path.exists(original_file):
        print("\n" + "=" * 50)
        print("原始数据集分析结果:")
        print("=" * 50)
        analyze_dataset(original_file)

    if os.path.exists(filtered_file):
        print("\n" + "=" * 50)
        print("过滤后数据集分析结果:")
        print("=" * 50)
        analyze_dataset(filtered_file)