import torch
import os
import json
import jieba
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_chinese import Rouge
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from metric_utils import compute_additional_metrics  # 导入刚才写的工具

# --- 配置区域 ---
BASE_MODEL_PATH = "/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "/root/autodl-tmp/output/qwen2_5-7b-medical-lora—pro/v0-20260110-211358/checkpoint-45" 
DB_PATH = "/root/autodl-tmp/chroma_db"
TEST_FILE = "/root/autodl-tmp/medical_sft_pro_test.jsonl"
OUTPUT_FILE = "lora_evaluation_report.json"

def load_models():
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    
    if LORA_PATH:
        print(f"✅ 挂载 LoRA 权重: {LORA_PATH}")
        model = PeftModel.from_pretrained(model, LORA_PATH)
    else:
        print("ℹ️ 使用纯基座模型进行评估")
        
    model.eval()
    
    print("正在加载向量数据库...")
    embeddings = HuggingFaceEmbeddings(
        model_name="/root/autodl-tmp/models/AI-ModelScope/bge-small-zh-v1.5",
        model_kwargs={'device': 'cuda'}
    )
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    return tokenizer, model, vector_db

def generate_response(model, tokenizer, question, context):
    prompt = f"""你是一个专业的医生。基于以下参考资料回答问题：
资料：{context}
问题：{question}
请给出结构化的回答（包括病情分析、指导建议、风险提示）。"""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_new_tokens=512, temperature=0.3)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def main():
    tokenizer, model, vector_db = load_models()
    rouge = Rouge()
    
    results = {
        "rouge-1": [], "rouge-2": [], "rouge-l": [],
        "accuracy": [], "citation_f1": [], "hallucination": []
    }
    
    # 读取测试集
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        data = data[:50] 

    print(f"🚀 开始评估 {len(data)} 条样本...")
    
    for item in tqdm(data):
        query = item['instruction']
        reference = item['output']
        
        # 1. RAG 检索
        docs = vector_db.similarity_search(query, k=3)
        context_text = "\n".join([d.page_content for d in docs])
        
        # 2. 模型生成
        prediction = generate_response(model, tokenizer, query, context_text)
        
        # 3. 计算 ROUGE
        prediction_seg = ' '.join(jieba.cut(prediction))
        reference_seg = ' '.join(jieba.cut(reference))
        try:
            scores = rouge.get_scores(prediction_seg, reference_seg)
            results['rouge-1'].append(scores[0]['rouge-1']['f'] * 100)
            results['rouge-2'].append(scores[0]['rouge-2']['f'] * 100)
            results['rouge-l'].append(scores[0]['rouge-l']['f'] * 100)
        except:
            pass 
            
        # 4. 计算高级指标
        adv_metrics = compute_additional_metrics(prediction, reference, context_text)
        results['accuracy'].append(adv_metrics['accuracy'])
        results['citation_f1'].append(adv_metrics['citation_f1'])
        results['hallucination'].append(adv_metrics['hallucination'])

    # 汇总报告
    final_report = {k: round(np.mean(v), 2) for k, v in results.items()}
    print("\n" + "="*40)
    print("📊 最终评估报告 (Average Scores)")
    print("="*40)
    print(json.dumps(final_report, indent=4, ensure_ascii=False))
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)
    print(f"✅ 报告已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()