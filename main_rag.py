import torch
import os
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 路径设置
BASE_PATH = "/root/autodl-tmp"
DB_PATH = os.path.join(BASE_PATH, "chroma_db")
MODEL_CACHE = os.path.join(BASE_PATH, "models")

# 2. 模型下载
print("正在通过 ModelScope 下载模型...")
# 下载 LLM
llm_model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir=MODEL_CACHE)
# 下载 Embedding 模型 (修复 OSError 的关键)
embed_model_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir=MODEL_CACHE)

# 3. 加载 LLM
print("正在加载 Qwen2.5-7B 到 RTX 5090...")
tokenizer = AutoTokenizer.from_pretrained(llm_model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    llm_model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)

# 4. 加载 Embedding (指向本地路径)
print("正在加载本地 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name=embed_model_dir, # 使用刚才下载好的本地路径
    model_kwargs={'device': 'cuda'}
)

# 5. 连接数据库
if not os.path.exists(DB_PATH):
    print(f"❌ 错误：在 {DB_PATH} 未找到数据库，请先运行 build_db.py")
    exit()
    
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 6. 推理 Pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)
llm = HuggingFacePipeline(pipeline=pipe)

# 7. LCEL RAG 链
template = """你是一个专业的中文医疗助手。请根据以下参考资料回答用户的问题。
资料库：
{context}

问题：{question}

回答："""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    test_query = "头痛恶心肌肉痛关节痛怎么回事？"
    print(f"\n🚀 启动检索问答...\n提问：{test_query}")
    try:
        response = rag_chain.invoke(test_query)
        print(f"\n✅ AI回答：\n{response}")
    except Exception as e:
        print(f"❌ 运行出错：{e}")