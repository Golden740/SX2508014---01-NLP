import torch.utils._pytree as pytree
from swift.llm import InferArguments, infer_main

# 1. 兼容性补丁 (针对你的 Torch 2.9)
if not hasattr(pytree, 'register_pytree_node'):
    pytree.register_pytree_node = pytree._register_pytree_node

# 2. 配置推理参数 (严格对齐 Swift 3.x 规范)
infer_args = InferArguments(
    # 基座模型路径
    model='/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct', 
    
    # 你微调生成的权重路径 (基于你 image_1534d7.png 的成功输出)
    ckpt_dir='/root/autodl-tmp/output/qwen2_5-7b-medical-lora/v0-20251230-233347/checkpoint-45', 
    
    # 必须指定模板
    template='qwen',
    
    
    # 生成控制
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)

if __name__ == "__main__":
    print("🩺 正在加载医学 LoRA 模型，准备进行对话测试...")
    # 启动交互式命令行界面
    infer_main(infer_args)