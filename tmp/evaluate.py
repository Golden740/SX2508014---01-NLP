import os
import json
import torch
import jieba
from tqdm import tqdm
from rouge_chinese import Rouge
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. 路径与配置 ---
BASE_PATH = "/root/autodl-tmp"
MODEL_PATH = os.path.join(BASE_PATH, "output/qwen2_5-7b-medical-lora/v0-20251230-233347/checkpoint-45-merged") 
DB_PATH = os.path.join(BASE_PATH, "chroma_db")
TEST_DATA_PATH = os.path.join(BASE_PATH, "medical_sft_pro_test.jsonl")

# 评估参数
SAMPLE_NUM = 50  
K_VALUE = 50     

print(f"正在初始化评估系统 (长上下文模式 K={K_VALUE})...")

# --- 2. 加载组件 ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True
)

embeddings = HuggingFaceEmbeddings(
    model_name=os.path.join(BASE_PATH, "models/AI-ModelScope/bge-small-zh-v1.5"),
    model_kwargs={'device': 'cuda'}
)
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": K_VALUE})

# --- 3. 评估核心逻辑 ---
def run_evaluation():
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ 错误：找不到测试集文件 {TEST_DATA_PATH}")
        return

    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_samples = [json.loads(line) for line in f][:SAMPLE_NUM]

    rouge = Rouge()
    preds, refs = [], []
    
    system_instruction = (
        "你是一个极度专业的医疗助手。请在参考资料的基础上，给出具有实操意义、条理清晰的医疗建议。\n\n"
        "【重要提示】：\n"
        "1. 参考资料已按相关性排序，请优先参考靠前的核心资料。\n"
        "2. 若资料量较大且包含无关干扰，请果断忽略，严禁产生幻觉。\n\n"
        "【回答策略】：\n"
        "1. 背景分析：简要说明症状可能的原因。\n"
        "2. 缓解方案：使用数字列表给出具体措施。\n"
        "3. 药物指导：提及常用非处方药并强调遵医嘱。\n"
        "4. 警示说明：提醒及时就医。\n"
        "5. 专业后缀：结尾固定包含“以上信息仅供参考，实际操作时请遵循专业医生的意见。”"
    )

    print(f"🚀 开始对 {SAMPLE_NUM} 条样本进行长上下文 RAG 评估...")

    for item in tqdm(test_samples):
        question = item.get('input', '')
        ground_truth = item.get('output', '')

        # a. 模拟长上下文检索 (>32k tokens 压力测试)
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        
        # b. 构造完整推理 Prompt
        prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n" \
                 f"<|im_start|>user\n参考资料内容：\n{context}\n\n用户咨询问题：{question}<|im_end|>\n" \
                 f"<|im_start|>assistant\n"
        
        # c. 模型生成
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=600, 
                temperature=0.3, 
                repetition_penalty=1.1
            )
        
        # d. 提取回复并清理
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_output.split("assistant")[-1].strip()
        
        # e. 分词以计算中文 Rouge
        preds.append(" ".join(jieba.cut(response)))
        refs.append(" ".join(jieba.cut(ground_truth)))

    # 4. 计算并输出结果
    scores = rouge.get_scores(preds, refs, avg=True)
    
    print("\n" + "="*50)
    print(f"📊 智医 RAG 评估报告 (测试模式: K={K_VALUE} 长上下文)")
    print("="*50)
    print(f"ROUGE-1 (单词覆盖率): {scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 (短语匹配度): {scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L (语义逻辑关联): {scores['rouge-l']['f']:.4f}")
    print("="*50)
    
    save_path = "lora_evaluation_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4)
    print(f"结果已保存至: {save_path}")

if __name__ == "__main__":
    run_evaluation()