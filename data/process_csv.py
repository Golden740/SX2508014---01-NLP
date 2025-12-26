import pandas as pd
import json
import os

# 1. 读取 CSV 文件
# 注意：如果报错 'UnicodeDecodeError'，请尝试把 encoding='utf-8' 改为 'gbk'
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
print(f"当前工作目录已切换至: {current_dir}")

print("正在读取 CSV 文件，这可能需要几秒钟...")
df_questions = pd.read_csv('question.csv', encoding='utf-8')
df_answers = pd.read_csv('answer.csv', encoding='utf-8')

print(f"原始问题数量: {len(df_questions)}")
print(f"原始回答数量: {len(df_answers)}")

# 2. 合并表格 (Merge)
# 相当于 SQL 里的 JOIN 操作，通过 'question_id' 把两个表连起来
# content_x 是问题，content_y 是回答
print("正在合并问答对...")
merged_df = pd.merge(df_questions, df_answers, on='question_id', how='inner')

# 3. 数据清洗
# 去掉空值
merged_df.dropna(subset=['content_x', 'content_y'], inplace=True)
# 过滤掉太短的问题或回答（小于5个字的可能是无效数据）
merged_df = merged_df[merged_df['content_x'].str.len() > 5]
merged_df = merged_df[merged_df['content_y'].str.len() > 10]

print(f"合并并清洗后的有效问答对数量: {len(merged_df)}")

# 4. 随机采样 (Sampling)
# 作业要求 5k 条，我们取 6000 条以防万一
# 如果数据量不够 6000，就取全部
sample_size = 6000
if len(merged_df) > sample_size:
    sampled_df = merged_df.sample(n=sample_size, random_state=42)
else:
    sampled_df = merged_df

print(f"已随机抽取 {len(sampled_df)} 条数据用于作业")

# 5. 格式化为 RAG 标准 JSON
# 目标结构：[{"page_content": "...", "metadata": {...}}, ...]
rag_data = []

for index, row in sampled_df.iterrows():
    q = row['content_x'] # 问题列
    a = row['content_y'] # 回答列
    q_id = row['question_id']
    
    # 拼接文本块：这是模型最终看到的内容
    text_chunk = f"用户问题：{q}\n医生建议：{a}"
    
    # 构建元数据：用于作业要求的“引用来源显示”
    # 在 RAG 检索回来时，你可以展示 "来源ID: xxx"
    meta = {
        "source_id": str(q_id),
        "dataset": "CMedQA2"
    }
    
    rag_data.append({
        "page_content": text_chunk,
        "metadata": meta
    })

# 6. 保存文件
output_file = 'rag_data_clean.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(rag_data, f, ensure_ascii=False, indent=2)

print(f"✅ 处理完成！文件已保存为: {output_file}")
print("你可以打开这个 JSON 文件检查一下内容。")