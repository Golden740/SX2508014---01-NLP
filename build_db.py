import json
import os
from modelscope import snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# --- 核心修复：1.x 版本的导入路径 ---
from langchain_core.documents import Document

# 1. 路径配置
BASE_PATH = "/root/autodl-tmp"
JSON_DATA = os.path.join(BASE_PATH, "rag_data_clean.json")
DB_PATH = os.path.join(BASE_PATH, "chroma_db")
MODEL_CACHE = os.path.join(BASE_PATH, "models")

# 2. 检查数据文件
if not os.path.exists(JSON_DATA):
    print(f"❌ 错误：未找到 {JSON_DATA}。请确保已上传 rag_data_clean.json")
    exit()

# 3. 下载/加载 Embedding 模型
print("正在通过 ModelScope 加载 Embedding 模型...")
embed_model_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir=MODEL_CACHE)

embeddings = HuggingFaceEmbeddings(
    model_name=embed_model_dir,
    model_kwargs={'device': 'cuda'}
)

# 4. 读取数据并转换为 Document 对象
print("正在读取并解析数据...")
with open(JSON_DATA, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 使用 1.x 的 Document 构造方式
documents = [
    Document(
        page_content=item['page_content'], 
        metadata=item.get('metadata', {})
    ) 
    for item in raw_data
]

# 5. 构建并持久化向量数据库
print(f"正在构建向量数据库 (Chroma)，存放至: {DB_PATH}...")
# 注意：Chroma 0.4.x 以后会自动持久化，不需要手动调用 .persist()
vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print(f"✅ 成功！已处理 {len(documents)} 条医疗问答数据。")