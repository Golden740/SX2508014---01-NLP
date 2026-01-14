import jieba
from collections import Counter

def compute_additional_metrics(response, reference, context):
    """
    计算扩展指标：准确率、引用F1、幻觉率
    
    :param response: 模型生成的回答
    :param reference: 标准答案 (Ground Truth)
    :param context: RAG检索到的参考资料文本
    """
    # 1. 预处理分词
    resp_tokens = set(jieba.lcut(response))
    ref_tokens = set(jieba.lcut(reference))
    ctx_tokens = set(jieba.lcut(context))
    
    # 移除停用词
    resp_tokens = {w for w in resp_tokens if len(w) > 1}
    ref_tokens = {w for w in ref_tokens if len(w) > 1}
    ctx_tokens = {w for w in ctx_tokens if len(w) > 1}

    # --- 指标 1: 准确率 ---
    # 定义：模型回答中包含了多少标准答案里的关键词
    if len(ref_tokens) == 0:
        accuracy = 0.0
    else:
        common_with_ref = resp_tokens.intersection(ref_tokens)
        accuracy = len(common_with_ref) / len(ref_tokens)

    # --- 指标 2: 引用 F1---
    # 定义：模型回答与检索到的上下文有多大的重合度
    if len(ctx_tokens) == 0 or len(resp_tokens) == 0:
        citation_f1 = 0.0
        precision = 0.0
    else:
        common_with_ctx = resp_tokens.intersection(ctx_tokens)
        precision = len(common_with_ctx) / len(resp_tokens) 
        recall = len(common_with_ctx) / len(ctx_tokens)     
        
        if (precision + recall) == 0:
            citation_f1 = 0.0
        else:
            citation_f1 = 2 * (precision * recall) / (precision + recall)

    # --- 指标 3: 幻觉率 ---
    # 定义：模型回答中有多少内容既不在参考资料里，也不在标准答案里（瞎编率）
    valid_knowledge = ctx_tokens.union(ref_tokens)
    if len(resp_tokens) == 0:
        hallucination_rate = 0.0
    else:
        supported_tokens = resp_tokens.intersection(valid_knowledge)
        hallucination_rate = 1.0 - (len(supported_tokens) / len(resp_tokens))

    return {
        "accuracy": round(accuracy * 100, 2),       
        "citation_f1": round(citation_f1 * 100, 2), 
        "hallucination": round(hallucination_rate * 100, 2) 
    }