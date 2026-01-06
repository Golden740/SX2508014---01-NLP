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

print("正在启动智医 RAG 系统 (结构化输出增强版)...")

# 下载并加载模型
llm_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir=MODEL_CACHE)
embed_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir=MODEL_CACHE)

tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    llm_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
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

        # 构造给模型看的全文（可能包含 32k tokens）
        context = "\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
        
        # 构造给用户看的简版资料溯源（仅展示前 5 条）
        source_display_content = "### 📚 检索到的参考资料\n\n"
        if not docs:
            source_display_content += "⚠️ 未在知识库中找到相关匹配内容。"
        
        for i, d in enumerate(docs[:5]): # 只取前 5 条展示
            source_display_content += f"### 📍 核心资料来源 {i+1}\n{d.page_content[:200]}...\n\n---\n"
        
        if len(docs) > 5:
            source_display_content += f"\n*注：后台已检索并分析其余 {len(docs)-5} 条辅助资料以确保结论准确。*"
        
        # b. 【高标准回复】集成结构化指令与知识扩展的 Prompt
        prompt = f"""<|im_start|>system
你是一个极度专业的医疗助手。请在参考资料的基础上，给出具有实操意义、条理清晰的医疗建议。

【重要提示】：
1. 参考资料已按相关性排序，请优先参考靠前的核心资料。
2. 若资料量较大且包含无关干扰，请果断忽略，严禁产生幻觉。

【回答策略】：
1. 背景分析：简要说明症状可能的原因。
2. 缓解方案：使用数字列表给出具体措施。
3. 药物指导：提及常用非处方药并强调遵医嘱。
4. 警示说明：提醒及时就医。
5. 专业后缀：结尾固定包含“以上信息仅供参考，实际操作时请遵循专业医生的意见。”<|im_end|>
<|im_start|>user
参考资料内容：
{context}

用户咨询问题：{message}<|im_end|>
<|im_start|>assistant
"""

        # c. 推理参数优化
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_kwargs = dict(
            inputs, 
            streamer=streamer, 
            max_new_tokens=600,     # 稍微调大字数上限，允许模型写出更详细的建议
            temperature=0.3,        # 稍微提升一点温度（从0.2到0.4），允许模型在专业范围内进行合理的语言润色
            repetition_penalty=1.2, # 降低惩罚力度，防止模型因为怕重复而不敢写出结构相似的建议
            top_p=0.9,
            do_sample=True
        )
        
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        # d. 实时流式响应
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