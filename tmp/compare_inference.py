import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# --- 1. 配置路径 ---
MODEL_PATH = "/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "/root/autodl-tmp/output/qwen2_5-7b-medical-lora—pro/v0-20260110-211358/checkpoint-45"

def get_response(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, temperature=0.7)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# --- 2. 加载基座模型 ---
print("正在加载基座模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

# --- 3. 进行对比推理 ---
question = "我经常反酸、烧心，尤其是晚上躺下后。请给出详细建议。"

print("\n" + "="*30 + " 基座模型回答 " + "="*30)
print(get_response(base_model, tokenizer, question))

# --- 4. 挂载 LoRA 补丁 ---
print("\n正在挂载 LoRA 补丁...")
lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("\n" + "="*30 + " 微调后模型回答 " + "="*30)
print(get_response(lora_model, tokenizer, question))