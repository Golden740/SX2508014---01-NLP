import pandas as pd
import json
import random
from sklearn.model_selection import train_test_split

# --- 配置 ---
RAG_JSON = "/root/autodl-tmp/rag_data_clean.json"
QUESTION_CSV = "/root/autodl-tmp/data/question.csv"
ANSWER_CSV = "/root/autodl-tmp/data/answer.csv"
TRAIN_FILE = "/root/autodl-tmp/medical_sft_pro_train.jsonl"
TEST_FILE = "/root/autodl-tmp/medical_sft_pro_test.jsonl"

def build_pro_dataset():
    # 1. 提取检索库 ID 对齐
    with open(RAG_JSON, 'r', encoding='utf-8') as f:
        rag_data = json.load(f)
    rag_ids = {item['question_id'] for item in rag_data if 'question_id' in item}

    # 2. 合并 CSV 提取正样本
    df_q = pd.read_csv(QUESTION_CSV)
    df_a = pd.read_csv(ANSWER_CSV)
    merged_df = pd.merge(df_q[df_q[df_q.columns[0]].isin(rag_ids)], df_a, left_on=df_q.columns[0], right_on='ans_id')

    # 3. 同步结构化 System Prompt
    system_instruction = (
    "你是一个极度专业的医疗助手。请在参考资料的基础上，给出具有实操意义、条理清晰的医疗建议。\n\n"
    "【重要提示】：\n"
    "1. 参考资料已按相关性从高到低排序，请优先参考排名靠前的资料。\n"
    "2. 若后半部分资料与用户问题无关或存在逻辑冲突，请果断忽略，严禁被无关信息误导产生幻觉。\n"
    "3. 即使资料库提供的条目较多（如 50 条），你也必须保持回复的精炼和结构化。\n\n"
    "【回答策略】：\n"
    "1. 背景分析：简要说明可能原因。\n"
    "2. 缓解方案：使用数字列表给出具体措施。\n"
    "3. 药物指导：提及常用非处方药并强调遵医嘱。\n"
    "4. 警示说明：提醒及时就医。\n"
    "5. 专业后缀：结尾固定包含“以上信息仅供参考，实际操作时请遵循专业医生的意见。”"
)

    sft_data = []
    for _, row in merged_df.iterrows():
        # 正确资料作为核心
        core_context = row['answer_content']
        
        # 随机抽取 2 条其他领域的医疗资料作为“噪声”拼在后面
        noise_samples = df_a.sample(2)['answer_content'].tolist()
        extended_context = f"【核心资料】：\n{core_context}\n\n【补充背景（可能无关）】：\n" + "\n".join(noise_samples)
        
        sft_data.append({
            "instruction": system_instruction,
            "input": f"参考资料内容：\n{extended_context}\n\n用户咨询问题：{row['content']}",
            "output": core_context # 期望输出依然只围绕核心资料，不被噪声干扰
        })

    # 4. 注入拒识样本 (强化“拒绝回答”逻辑)
    reject_ans = "抱歉，您咨询的问题超出了我的医疗知识库范围。为了您的健康，建议咨询相关专科医生。"
    for _ in range(250):
        sft_data.append({
            "instruction": system_instruction,
            "input": f"参考资料内容：\n无相关资料\n\n用户咨询问题：{random.choice(['如何写代码', '推荐电影', '秦始皇是谁'])}",
            "output": reject_ans
        })

    # 5. 保存
    random.shuffle(sft_data)
    train, test = train_test_split(sft_data, test_size=0.1, random_state=42)
    for path, data in [(TRAIN_FILE, train), (TEST_FILE, test)]:
        with open(path, 'w', encoding='utf-8') as f:
            for item in data: f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ 构建完成。训练集: {len(train)}，测试集: {len(test)}")

if __name__ == "__main__":
    build_pro_dataset()