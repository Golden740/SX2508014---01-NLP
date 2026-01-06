import torch.utils._pytree as pytree
from swift.llm import InferArguments, infer_main

# 1. 核心兼容补丁
if not hasattr(pytree, 'register_pytree_node'):
    pytree.register_pytree_node = pytree._register_pytree_node

# 2. 配置推理参数
# 注意：Swift 3.x 会自动根据 ckpt_dir 识别配置，不需要 load_dataset_config
infer_args = InferArguments(
    model='/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct',
    ckpt_dir='/root/autodl-tmp/output/qwen2_5-7b-medical-lora/v0-20251230-233347/checkpoint-45',
    template='qwen',
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)

if __name__ == "__main__":
    print("🚀 正在启动交互式医学问答界面...")
    # 只要这行跑通，你就能直接在终端跟模型聊天了
    infer_main(infer_args)