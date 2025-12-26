import gradio as gr
import torch
import os
import gc
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. 资源与路径初始化 ---
BASE_PATH = "/root/autodl-tmp"
DB_PATH = os.path.join(BASE_PATH, "chroma_db")
MODEL_CACHE = os.path.join(BASE_PATH, "models")

print("正在初始化智医 RAG 系统 (精简回复版)...")

# 下载/加载模型
llm_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir=MODEL_CACHE)
embed_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir=MODEL_CACHE)

# 加载 Tokenizer 和 Model
tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    llm_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)

# 加载 RAG 检索器
embeddings = HuggingFaceEmbeddings(model_name=embed_dir, model_kwargs={'device': 'cuda'})
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# --- 2. 核心推理逻辑 ---
def chat_and_retrieve(message):
    try:
        # 显存清理
        torch.cuda.empty_cache()
        gc.collect()

        # a. 检索相关文档
        docs = retriever.invoke(message)
        source_content = ""
        context_parts = []
        for i, d in enumerate(docs):
            source_content += f"### 📍 资料来源 {i+1}\n{d.page_content}\n\n---\n"
            context_parts.append(d.page_content)
        
        context = "\n\n".join(context_parts)
        
        # b. 构造严格的 ChatML 格式 Prompt
        # 这种格式能有效防止 Qwen 模型复述 Prompt 里的资料标签
        prompt = f"""<|im_start|>system
你是一个专业的医疗助手。请根据提供的资料进行总结回答。
要求：
- 直接给出医疗建议或结论，禁止原样复述参考资料中的“用户问题”或“医生建议”。
- 语言精炼，禁止出现重复的段落。
- 若资料无关，请基于专业医学知识回答并提示建议仅供参考。<|im_end|>
<|im_start|>user
参考资料内容：
{context}

用户咨询问题：{message}<|im_end|>
<|im_start|>assistant
"""

        # c. 准备推理输入
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # d. 生成参数优化 (关键：低温度 + 重复惩罚)
        gen_kwargs = dict(
            inputs, 
            streamer=streamer, 
            max_new_tokens=512, 
            temperature=0.3,      # 降低随机性
            repetition_penalty=1.2, # 惩罚重复内容
            top_p=0.8
        )
        
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        # e. 流式迭代输出
        full_response = ""
        for new_text in streamer:
            # 过滤可能出现的停止符
            clean_text = new_text.replace("<|im_end|>", "").replace("<|im_start|>", "")
            full_response += clean_text
            # 仅向前端返回模型生成的纯净回答和溯源资料
            yield full_response.strip(), source_content
            
    except Exception as e:
        yield f"⚠️ 系统繁忙: {str(e)}", "检索失败"

# --- 3. Gradio 界面设计 (适配字典格式校验) ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), title="智医 RAG") as demo:
    gr.Markdown("# 🏥 智医 RAG：医疗问答平台")
    gr.Markdown("提示：系统会自动从知识库检索相关病例并由 AI 进行总结回复。")
    
    with gr.Row():
        # 对话区
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="医生助手对话框", height=550)
            msg = gr.Textbox(label="描述您的症状或问题", placeholder="输入后按回车发送...")
            with gr.Row():
                submit_btn = gr.Button("🚀 提交咨询", variant="primary")
                clear_btn = gr.Button("🗑️ 重置对话")

        # 资料区
        with gr.Column(scale=3):
            gr.Markdown("### 🔍 知识库检索溯源")
            sources_display = gr.Markdown("等待提问以显示参考资料...", label="参考资料")

    # --- 交互绑定 ---
    def respond(user_message, chat_history):
        if chat_history is None:
            chat_history = []
            
        # 封装为新版字典格式
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": ""})
        
        # 获取生成器输出
        response_gen = chat_and_retrieve(user_message)
        
        for chat_text, source_text in response_gen:
            chat_history[-1]["content"] = chat_text
            yield chat_history, source_text

    # 事件流
    submit_btn.click(respond, [msg, chatbot], [chatbot, sources_display]).then(lambda: "", None, [msg])
    msg.submit(respond, [msg, chatbot], [chatbot, sources_display]).then(lambda: "", None, [msg])
    clear_btn.click(lambda: (None, "等待检索..."), None, [chatbot, sources_display])

if __name__ == "__main__":
    # 关闭 share=True 以避开 frpc 报错，使用 AutoDL 自带映射访问
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)