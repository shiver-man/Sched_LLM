import matplotlib.pyplot as plt
import os
from pathlib import Path

def recreate_image_plot():
    # 设置字号为 12pt (小四)
    FONT_SIZE = 12.0
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
        'figure.titlesize': FONT_SIZE + 2
    })

    # 数据准备 (基于图片观测值)
    labels = ['Traditional(best)', 'Framework(GA)']
    makespan_data = [128.0, 119.0]
    transport_data = [10.0, 10.3]

    # 创建并列子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- 左图: Makespan Comparison ---
    axes[0].bar(labels, makespan_data, color='#1f77b4', width=0.7)
    axes[0].set_title('Makespan Comparison')
    axes[0].set_ylabel('Makespan')
    axes[0].set_ylim(0, 135)
    axes[0].grid(axis='y', linestyle='-', alpha=0.15)
    # 旋转标签以匹配图片
    plt.setp(axes[0].get_xticklabels(), rotation=15, ha='right')

    # --- 右图: Transport Time Comparison ---
    axes[1].bar(labels, transport_data, color='#1f77b4', width=0.7)
    axes[1].set_title('Transport Time Comparison')
    axes[1].set_ylabel('Transport Time')
    axes[1].set_ylim(0, 11)
    axes[1].grid(axis='y', linestyle='-', alpha=0.15)
    plt.setp(axes[1].get_xticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    
    # 保存路径
    out_dir = Path(r"c:\Users\shiver\Desktop\Sched_LLM\backend\data\fjsp_epl_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "recreated_comparison_12pt.svg"
    
    fig.savefig(out_file, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Plot recreated and saved to: {out_file}")

if __name__ == "__main__":
    recreate_image_plot()
