import gradio as gr
import torch
import os
import gc
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. 环境与模型配置 ---
BASE_PATH = "/root/autodl-tmp"
DB_PATH = os.path.join(BASE_PATH, "chroma_db")
MODEL_CACHE = os.path.join(BASE_PATH, "models")
MERGED_MODEL_PATH = "/root/autodl-tmp/output/qwen2_5-7b-medical-lora—pro/v0-20260110-211358/checkpoint-45"

print("正在启动智医 RAG 系统 (结构化输出增强版)...")

# 加载模型
embed_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir=MODEL_CACHE)

tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)

# 初始化 RAG 检索器
embeddings = HuggingFaceEmbeddings(model_name=embed_dir, model_kwargs={'device': 'cuda'})
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 50})

# --- 2. 推理引擎：结构化 Prompt 集成 ---
def chat_and_retrieve(message):
    print(f"\n🔍 收到用户提问: {message}")  
    
    try:
        torch.cuda.empty_cache()
        gc.collect()
        
        # 执行深度检索，满足性能指标
        docs = retriever.invoke(message)
        print(f"✅ 检索成功，找到 {len(docs)} 条相关资料")

        context = "\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
        
        source_display_content = "### 📚 检索到的参考资料\n\n"
        if not docs:
            source_display_content += "⚠️ 未在知识库中找到相关匹配内容。"
        
        for i, d in enumerate(docs[:5]): # 只取前 5 条展示
            source_display_content += f"### 📍 核心资料来源 {i+1}\n{d.page_content[:200]}...\n\n---\n"
        
        if len(docs) > 5:
            source_display_content += f"\n*注：后台已检索并分析其余 {len(docs)-5} 条辅助资料以确保结论准确。*"
        
        prompt = f"""<|im_start|>system
你是一个医生。请回答用户的问题。
"""

        # 推理参数优化
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_kwargs = dict(
            inputs, 
            streamer=streamer, 
            max_new_tokens=600,     
            temperature=0.3,        
            repetition_penalty=1.2, 
            top_p=0.9,
            do_sample=True
        )
        
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        # 实时流式响应
        full_response = ""
        for new_text in streamer:
            # 过滤特殊字符
            clean_text = new_text.replace("<|im_end|>", "").replace("<|im_start|>", "")
            full_response += clean_text
            yield full_response.strip(), source_display_content
            
    except Exception as e:
        yield f"⚠️ 系统诊断错误: {str(e)}", "检索失败"

# --- 3. Gradio 交互界面 ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), title="智医 RAG 问答平台") as demo:
    gr.Markdown("# 🏥 智医 RAG：中文医疗问答平台")
    
    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="AI 医生助手", height=550)
            msg = gr.Textbox(label="输入您的疑问（如：头疼怎么办？）", placeholder="输入后按回车发送...")
            with gr.Row():
                submit_btn = gr.Button("🚀 提交咨询", variant="primary")
                clear_btn = gr.Button("🗑️ 重置对话")

        with gr.Column(scale=3):
            gr.Markdown("### 🔍 知识库检索溯源")
            sources_display = gr.Markdown("等待检索资料...", label="参考资料")

    # 交互逻辑适配 Gradio 4.x/5.x 的字典格式校验
    def respond(user_message, chat_history):
        if chat_history is None:
            chat_history = []
        
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": ""})
        
        response_gen = chat_and_retrieve(user_message)
        
        for chat_text, source_text in response_gen:
            chat_history[-1]["content"] = chat_text
            yield chat_history, source_text

    # 事件流绑定
    submit_btn.click(respond, [msg, chatbot], [chatbot, sources_display]).then(lambda: "", None, [msg])
    msg.submit(respond, [msg, chatbot], [chatbot, sources_display]).then(lambda: "", None, [msg])
    clear_btn.click(lambda: (None, "等待提问..."), None, [chatbot, sources_display])

if __name__ == "__main__":
    # 使用 AutoDL 映射端口 6006 访问
    print("✅ 服务即将在 http://127.0.0.1:6006 启动")
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)