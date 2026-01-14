import json
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 读取数据 ---
def load_metrics(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [
        data['rouge-1']['f'] * 100,
        data['rouge-2']['f'] * 100,
        data['rouge-l']['f'] * 100
    ]

base_scores = load_metrics('evaluation_results.json')
lora_scores = load_metrics('lora_evaluation_results.json')

labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
x = np.arange(len(labels))
width = 0.35

# --- 2. 绘图 ---
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

# 绘制柱状图
rects1 = ax.bar(x - width/2, base_scores, width, label='Base Model (Qwen2.5)', color='#bdc3c7', edgecolor='white', linewidth=1)
rects2 = ax.bar(x + width/2, lora_scores, width, label='Medical SFT Model (Ours)', color='#2ecc71', edgecolor='white', linewidth=1)

ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: Base vs. Fine-tuned', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 100) 
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 自动标注数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('performance_comparison.png')
print("✅ 对比柱状图已生成：performance_comparison.png")