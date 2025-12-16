# filter_sft_data.py - 筛选sft_train_direct.jsonl
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

def parse_sft_record(item):
    """解析SFT记录的各个字段"""
    input_text = item.get('input', '')
    output_text = item.get('output', '')
    
    # 从output提取信息
    # 格式: **错误类型**: xxx
    # **改正**: xxx
    # **解释**: xxx
    type_match = re.search(r'\*\*错误类型\*\*:\s*(.+)', output_text)
    correct_match = re.search(r'\*\*改正\*\*:\s*(.+)', output_text)
    explain_match = re.search(r'\*\*解释\*\*:\s*(.+)', output_text, re.DOTALL)
    
    return {
        'error_sentence': input_text,
        'correct_sentence': correct_match.group(1).strip() if correct_match else "",
        'error_type': type_match.group(1).strip() if type_match else "",
        'explanation': explain_match.group(1).strip() if explain_match else ""
    }

def auto_filter(items):
    """自动过滤修正前后句子相同的情况"""
    filtered_items = []
    skipped_count = 0
    
    for item in items:
        try:
            record = parse_sft_record(item)
            # 检查修正前后句子是否相同
            if record['error_sentence'].strip().lower() == record['correct_sentence'].strip().lower():
                skipped_count += 1
                continue
            filtered_items.append(item)
        except Exception as e:
            print(f"自动过滤时解析失败: {e}")
            skipped_count += 1
    
    print(f"自动过滤完成: 跳过 {skipped_count} 条修正前后句子相同的记录")
    return filtered_items

def validate_record(item):
    """验证单条记录的基本格式"""
    if not isinstance(item, dict):
        return False, "不是有效的JSON对象"
    
    if 'input' not in item or 'output' not in item:
        return False, "缺少必要字段(input或output)"
    
    input_text = item['input']
    output_text = item['output']
    
    if not input_text or not output_text:
        return False, "input或output为空"
    
    # 检查输出格式是否正确
    required_patterns = [
        r'\*\*错误类型\*\*:',
        r'\*\*改正\*\*:',
        r'\*\*解释\*\*:'
    ]
    
    for pattern in required_patterns:
        if not re.search(pattern, output_text):
            # 提取字段名称，如从r'\*\*错误类型\*\*:'中提取'错误类型'
            field_name = re.search(r'\*\*(.+?)\*\*:', pattern).group(1) if re.search(r'\*\*(.+?)\*\*:', pattern) else '未知字段'
            return False, f"输出格式不正确，缺少 {field_name} 字段"
    
    # 额外验证：检查改正是否合理
    record = parse_sft_record(item)
    if record['error_sentence'].strip().lower() == record['correct_sentence'].strip().lower():
        return False, "修正前后句子相同，无实际修正"
    
    return True, "格式正确"

def improve_sample(sample):
    """使用API改进评分低的样本"""
    try:
        record = parse_sft_record(sample)
        
        # 首先检查原样本是否真的需要改进
        if record['error_sentence'].strip().lower() == record['correct_sentence'].strip().lower():
            print(f"  ⚠️  原样本已经是正确的，无需改进")
            return None
        
        prompt = f"""
        请严格改进以下英语语法纠错样本，确保：
        1. 错误类型标注准确
        2. 改正后的句子必须与原句有实质性不同（不能完全相同）
        3. 解释清晰、准确且符合语法规则
        4. 确保原句确实存在语法错误
        5. 保持输出格式不变
        
        当前样本：
        原句：{record['error_sentence']}
        错误类型：{record['error_type']}
        改正：{record['correct_sentence']}
        解释：{record['explanation']}
        
        请以相同格式返回改进后的样本：
        **错误类型**: [改进后的错误类型]
        **改正**: [改进后的正确句子]
        **解释**: [改进后的清晰解释]
        """
        
        response = client.chat.completions.create(
            model="kimi-k2-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=100
        )
        
        improved_output = response.choices[0].message.content
        improved_item = sample.copy()
        improved_item['output'] = improved_output
        
        # 预先检查改进结果
        improved_record = parse_sft_record(improved_item)
        if improved_record['error_sentence'].strip().lower() == improved_record['correct_sentence'].strip().lower():
            print(f"  ⚠️  API返回的改进结果与原句相同，跳过该样本")
            return None
        
        return improved_item
    except Exception as e:
        print(f"改进样本失败: {e}")
        return None

def evaluate_batch(items):
    """批量评估10条SFT数据"""
    # 构建评估用的数据列表
    eval_items = [parse_sft_record(item) for item in items]
    
    prompt = f"""
    请批量评估以下英语语法纠错数据的质量，每一项从12个维度打分（0-5分）：
    
    评分标准：
    1. **错误真实性**：是否真实反映中国学生常见错误？（0=刻意编造，5=非常自然）
    2. **改正正确性**：改正后句子是否完全正确？（0=仍有严重错误，5=完美正确）
    3. **类型准确性**：error_type标注是否准确描述了错误类型？（0=错误标注，5=精确匹配）
    4. **解释清晰度**：explanation是否正确解释了错误原因且与错误句子相关？（0=错误或不相关，5=清晰准确）
    5. **格式正确**：error_sentence和correct_sentence中是否仅包含英文？（0=包含中文或其他非英文，5=纯英文）
    6. **必要性**：将error_sentence改成correct_sentence是否有实际提升？是否满足正常英语逻辑？（0=无必要或逻辑不通，5=非常必要且逻辑通顺）
    7. **自然度**：error_sentence和correct_sentence是否符合英语表达习惯？（0=不自然，5=非常自然）
    8. **解释合理性**：explanation是否与error_type和具体错误匹配？是否符合语法规则？（0=解释与错误不符或语法知识错误，5=解释完全匹配且语法知识正确）
    9. **无错误句子修改**：error_sentence是否确实存在语法错误？（0=原句正确却被修改，5=原句确实有语法错误）
    10. **错误类型一致性**：error_type是否与实际错误性质一致？（0=错误类型完全错误，5=错误类型准确）
    11. **词汇准确性**：是否存在词汇使用错误？（0=严重词汇错误，5=词汇使用完全正确）
    12. **句子结构合理性**：句子结构是否符合英语语法规则？（0=结构严重错误，5=结构完美）
    
    要求：
    1. 只有当某条数据的所有12个维度评分均大于等于4分时，才算合格
    2. 特别注意检查：
       - 是否有正确句子被错误修改（原句是正确的，却被错误地标注为错误句子）
       - 是否有解释与错误类型不匹配的情况
       - 是否有语法知识错误的解释
       - 是否有词汇使用错误
       - 是否有句子结构错误
    3. 输出格式必须严格遵循：
       ```
       合格索引: 0,3,9
       需要改进的索引: 1,4,6,7,8
       不保留数据的索引: 2,5
       ```
       其中：
       - 合格索引：所有维度评分均≥4分的样本
       - 需要改进的索引：有改进空间但不是完全错误的样本
       - 不保留数据的索引：原句本身就是正确的，却被错误标注为错误句子的样本（第9维度评分≤1的样本）
       （如果没有对应项，相应部分留空）
    4. 请认真检查每个维度，确保评分准确
    
    待评估数据：
    {json.dumps(eval_items, ensure_ascii=False, indent=2)}
    """
    
    try:
        response = client.chat.completions.create(
            model="kimi-k2-0905-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=120  # 增加超时时间以处理更复杂的评估
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"评估失败: {e}")
        return None

def validate_improvement(original_item, improved_item):
    """验证改进后的样本质量"""
    try:
        # 检查格式是否正确
        is_valid, message = validate_record(improved_item)
        if not is_valid:
            return False, f"格式验证失败: {message}"
        
        # 检查改进是否有效
        original_record = parse_sft_record(original_item)
        improved_record = parse_sft_record(improved_item)
        
        # 检查修正前后句子是否不同
        if improved_record['error_sentence'].strip().lower() == improved_record['correct_sentence'].strip().lower():
            return False, "改进后修正前后句子仍然相同"
        
        # 简单验证解释是否更合理
        if len(improved_record['explanation']) < 10:
            return False, "改进后解释仍然过于简单"
        
        return True, "改进有效"
    except Exception as e:
        return False, f"验证失败: {e}"

def analyze_errors(items):
    """分析过滤掉的数据，统计错误类型"""
    error_analysis = {
        "correct_sentences_modified": 0,
        "wrong_error_types": 0,
        "wrong_explanations": 0,
        "same_before_after": 0,
        "other_issues": 0
    }
    
    for item in items:
        try:
            record = parse_sft_record(item)
            # 检查是否修改了正确句子
            if record['error_sentence'].strip().lower() == record['correct_sentence'].strip().lower():
                error_analysis['same_before_after'] += 1
                continue
            
            # 这里可以添加更详细的错误分析逻辑
            # 暂时只做简单统计
        except Exception as e:
            error_analysis['other_issues'] += 1
    
    return error_analysis

def main():
    # 读取sft_train_direct.jsonl
    try:
        with open('sft_train_direct.jsonl', 'r', encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f]
        print(f"成功读取 {len(raw_data)} 条SFT数据")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 先进行基本格式验证
    valid_data = []
    invalid_count = 0
    
    for i, item in enumerate(raw_data):
        is_valid, message = validate_record(item)
        if is_valid:
            valid_data.append(item)
        else:
            invalid_count += 1
            if invalid_count <= 10:  # 只显示前10个错误示例
                print(f"记录 {i+1} 格式无效: {message}")
    
    if invalid_count > 10:
        print(f"... 还有 {invalid_count - 10} 条记录格式无效")
    
    print(f"格式验证完成: 有效记录 {len(valid_data)} 条, 无效记录 {invalid_count} 条")
    
    if not valid_data:
        print("没有有效记录可评估")
        return
    
    print(f"开始评估 {len(valid_data)} 条有效SFT数据...")
    
    # 批量处理（每批10条）
    batch_size = 10
    qualified_indices = []
    discard_indices = []  # 存储不保留数据的索引
    improved_items = []  # 存储改进后的样本
    total_batches = (len(valid_data) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(valid_data), batch_size):
        batch = valid_data[batch_idx:batch_idx+batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        print(f"\n批处理 {batch_num}/{total_batches}: 评估第 {batch_idx+1}-{min(batch_idx+batch_size, len(valid_data))} 条...")
        
        result = evaluate_batch(batch)
        if result:
            print(f"  评估结果: {result}")
            
            # 解析合格索引
            try:
                if "合格索引:" in result:
                    indices_str = result.split("合格索引:")[1].split("\n")[0].strip()
                    if indices_str:
                        batch_qualified = [int(x.strip()) for x in indices_str.split(",") if x.strip()]
                        # 转换为全局索引
                        qualified_indices.extend([batch_idx + idx for idx in batch_qualified])
                        print(f"  本批合格: {len(batch_qualified)} 条")
                    else:
                        print(f"  本批无合格数据")
                else:
                    print(f"  未找到合格索引")
            except Exception as e:
                print(f"  解析合格索引失败: {e}")
                # 不打印完整结果以避免过多输出
                print(f"  结果格式异常，跳过索引解析")
            
            # 解析需要改进的索引（如果有）
            try:
                if "需要改进的索引:" in result:
                    indices_str = result.split("需要改进的索引:")[1].split("\n")[0].strip()
                    if indices_str:
                        batch_improvement = [int(x.strip()) for x in indices_str.split(",") if x.strip()]
                        print(f"  本批需要改进: {len(batch_improvement)} 条")
                        
                        # 对需要改进的样本进行API改进
                        for idx in batch_improvement:
                            sample = batch[idx]
                            print(f"  正在改进样本 {batch_idx + idx + 1}...")
                            improved_sample = improve_sample(sample)
                            if improved_sample:
                                # 验证改进后的样本
                                is_valid, message = validate_improvement(sample, improved_sample)
                                if is_valid:
                                    improved_items.append(improved_sample)
                                    print(f"  ✅ 样本 {batch_idx + idx + 1} 改进成功")
                                else:
                                    print(f"  ❌ 样本 {batch_idx + idx + 1} 改进验证失败: {message}")
                            else:
                                print(f"  ❌ 样本 {batch_idx + idx + 1} 改进失败")
                            time.sleep(1.0)  # 避免API限速
            except Exception as e:
                print(f"  解析需要改进的索引失败: {e}")
        else:
            print(f"  评估失败，跳过本批")
        
        time.sleep(1.5)  # 避免API限速
    
    # 筛选合格数据
    filtered = [valid_data[idx] for idx in sorted(qualified_indices)]
    
    print(f"\n" + "="*60)
    print(f"筛选结果统计:")
    print(f"- 总记录数: {len(raw_data)}")
    print(f"- 有效记录数: {len(valid_data)}")
    print(f"- 合格记录数: {len(filtered)}")
    print(f"- 需要改进的记录数: {len(improved_items)} (改进成功)")
    print(f"- 不保留数据的记录数: {len(discard_indices)}")
    print(f"- 总保留率: {(len(filtered) + len(improved_items)) / len(raw_data) * 100:.1f}%")
    print(f"- 有效记录保留率: {(len(filtered) + len(improved_items)) / len(valid_data) * 100:.1f}%")
    print("="*60)
    
    # 合并合格和改进的样本
    all_valid_data = filtered + improved_items
    
    # 保存所有有效数据
    try:
        with open('sft_train_filtered.jsonl', 'w', encoding='utf-8') as f:
            for item in all_valid_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ 已保存 {len(all_valid_data)} 条数据到 sft_train_filtered.jsonl")
    except Exception as e:
        print(f"保存文件失败: {e}")
    
    # 显示几条示例
    if filtered:
        print("\n=== 合格数据示例 ===")
        for i, item in enumerate(filtered[:3]):
            print(f"\n第 {i+1} 条:")
            print(f"输入: {item['input']}")
            print(f"输出:")
            for line in item['output'].split('\n'):
                print(f"  {line}")
    
    # 分析错误原因
    error_analysis = analyze_errors(raw_data)
    print(f"\n=== 错误分析 ===")
    print(f"- 被错误修改的正确句子: {error_analysis['correct_sentences_modified']} 条")
    print(f"- 错误类型标注: {error_analysis['wrong_error_types']} 条")
    print(f"- 错误的解释内容: {error_analysis['wrong_explanations']} 条")
    print(f"- 其他问题: {error_analysis['other_issues']} 条")

if __name__ == "__main__":
    main()