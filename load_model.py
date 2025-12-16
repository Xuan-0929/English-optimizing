import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 强制4-bit配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("正在加载 Qwen2.5-3B-Instruct (4-bit量化版)...")

# 关键1: 强制所有参数上GPU
# 关键2: max_memory限制防止offload
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    cache_dir="models",
    device_map="auto",
    max_memory={0: "7GB"},  # 限制使用7GB，留1GB给训练动态分配
    quantization_config=bnb_config,
    trust_remote_code=True,
    dtype=torch.float16,
    offload_folder=None,  # 禁用CPU offloading
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    trust_remote_code=True,
    cache_dir="models"
)

# 保存tokenizer到本地模型目录
tokenizer.save_pretrained(
    "c:/Users/ASUS/Desktop/English optimizing/models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
)

print("\n✅ tokenizer已保存到本地模型目录")

# 验证：所有参数必须在GPU
device_counts = {}
for param in model.parameters():
    device = param.device
    device_counts[device] = device_counts.get(device, 0) + 1

print(f"\n✅ 参数分布: {device_counts}")
print(f"📊 显存占用: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

# 应该只有: {device(type='cuda', index=0): 434} (所有参数)
