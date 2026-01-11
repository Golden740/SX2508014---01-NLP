import json
import os
import glob
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 配置区：请确认这个路径是你训练输出的总目录
# 根据你之前上传的代码，应该是这个：
LOG_SEARCH_PATH = '/root/autodl-tmp/output/qwen2_5-7b-medical-lora—pro'
# ---------------------------------------------------------

def start_plotting():
    print(f"🔍 正在从 {LOG_SEARCH_PATH} 寻找训练日志...")
    
    # 递归搜索所有的 trainer_state.json
    state_files = glob.glob(os.path.join(LOG_SEARCH_PATH, "**/trainer_state.json"), recursive=True)
    
    if not state_files:
        print("❌ 错误：没找到日志文件！请确认你的训练是否已经生成了 output 文件夹。")
        return

    # 找到最新修改的那个日志文件
    latest_state_file = max(state_files, key=os.path.getmtime)
    print(f"✅ 找到最新日志: {latest_state_file}")

    with open(latest_state_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    history = data.get('log_history', [])
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for entry in history:
        # 提取训练 Loss
        if 'loss' in entry and 'step' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])
        # 提取验证 Loss (如果你原来的脚本开启了验证)
        if 'eval_loss' in entry and 'step' in entry:
            eval_steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])

    if not train_steps:
        print("⚠️ 日志文件中没有 Loss 数据，可能是训练步数太少还没触发 logging_steps。")
        return

    # --- 绘图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, color='#1f77b4', label='Training Loss', linewidth=2)
    
    if eval_steps:
        plt.plot(eval_steps, eval_loss, color='#ff7f0e', label='Validation Loss', linestyle='--', marker='o')
    
    plt.title('Fine-tuning Training Curve', fontsize=14)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 保存图片到根目录
    save_name = 'medical_training_plot.png'
    plt.savefig(save_name, dpi=300)
    print(f"\n✨✨✨ 恭喜！可视化图表已生成：{save_name}")
    print(f"提示：请在左侧文件树中找到 {save_name}，双击或右键下载即可。")

if __name__ == "__main__":
    start_plotting()