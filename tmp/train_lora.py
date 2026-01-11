import torch.utils._pytree as pytree
import os

# 1. 核心补丁 (针对 Torch 2.9)
if not hasattr(pytree, 'register_pytree_node'):
    pytree.register_pytree_node = pytree._register_pytree_node

# 2. 导入 Swift 3.x 训练类
from swift.llm import TrainArguments, sft_main

# 3. 配置训练参数 (严格对齐 Swift 3.x 规范)
sft_args = TrainArguments(
    # --- 模型与路径 ---
    model='/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct',
    train_type='lora',
    template='qwen',
    
    # --- 数据集 ---
    dataset=['/root/autodl-tmp/medical_sft_pro_train.jsonl'],
    val_dataset=['/root/autodl-tmp/medical_sft_pro_test.jsonl'],
    
    # --- 显存与计算 ---
    max_length=2048,
    gradient_checkpointing=True,
    
    # ✅ 修正报错：Swift 3.x 将 batch_size 拆分为以下两个参数
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1,
    
    # --- LoRA 核心参数 (已修正名) ---
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'], # 删掉 lora_ 前缀
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.05,

    # --- 🚀 核心修改：增加以下三行 ---
    eval_steps=5,               # 2. 每 10 步就跑一次验证集（建议与 logging_steps 一致，这样点最密集）
    logging_steps=5,            # 3. 每 10 步记录一次训练 Loss
    
    # --- 训练策略 ---
    learning_rate=1e-4,
    num_train_epochs=3,
    output_dir='output/qwen2_5-7b-medical-lora—pro',
    
    # --- 日志与保存 ---
    save_steps=50,
)

if __name__ == "__main__":
    print("🚀 正在启动 Swift 3.x 训练引擎...")
    sft_main(sft_args)