import matplotlib.pyplot as plt
import os
from pathlib import Path

def setup_style():
    FONT_SIZE = 12.0
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
        'figure.titlesize': FONT_SIZE + 2,
        'font.sans-serif': ['Arial', 'SimHei'] # 支持中文
    })

def plot_combined_gantt(out_path):
    """还原第一张图：框架(GA)的生产与运输甘特图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'T1': '#1f77b4', 'T2': '#ff7f0e', 'T3': '#2ca02c', 'T4': '#d62728'}
    
    # --- 左图: Production Gantt ---
    # Data: (start, duration)
    ax1.broken_barh([(0, 36)], (0.6, 0.8), facecolors=colors['T1'], edgecolors='black', linewidth=0.5)
    ax1.text(18, 1, 'T1', color='white', ha='center', va='center')
    
    ax1.broken_barh([(5, 21)], (1.6, 0.8), facecolors=colors['T2'], edgecolors='black', linewidth=0.5)
    ax1.text(15.5, 2, 'T2', color='white', ha='center', va='center')
    
    ax1.broken_barh([(38, 36)], (0.6, 0.8), facecolors=colors['T3'], edgecolors='black', linewidth=0.5)
    ax1.text(56, 1, 'T3', color='white', ha='center', va='center')
    
    ax1.broken_barh([(76, 43)], (0.6, 0.8), facecolors=colors['T4'], edgecolors='black', linewidth=0.5)
    ax1.text(97.5, 1, 'T4', color='white', ha='center', va='center')
    
    ax1.set_yticks([1, 2])
    ax1.set_yticklabels(['y-005', 'y-002'])
    ax1.set_title('Production Gantt (Best=GA)')
    ax1.set_xlabel('Time')
    ax1.grid(axis='x', linestyle='-', alpha=0.1)

    # --- 右图: Transport Gantt ---
    ax2.broken_barh([(0, 2)], (0.6, 0.8), facecolors=colors['T1'], edgecolors='black', linewidth=0.5)
    ax2.text(1, 1, 'T1', color='white', ha='center', va='center', fontsize=10)
    
    ax2.broken_barh([(0, 6)], (1.6, 0.8), facecolors=colors['T2'], edgecolors='black', linewidth=0.5)
    ax2.text(3, 2, 'T2', color='white', ha='center', va='center', fontsize=10)
    
    ax2.broken_barh([(36, 5)], (1.6, 0.8), facecolors=colors['T3'], edgecolors='black', linewidth=0.5)
    ax2.text(38.5, 2, 'T3', color='white', ha='center', va='center', fontsize=10)
    
    ax2.broken_barh([(74, 3)], (1.6, 0.8), facecolors=colors['T4'], edgecolors='black', linewidth=0.5)
    ax2.text(75.5, 2, 'T4', color='white', ha='center', va='center', fontsize=10)
    
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(['v2', 'v1'])
    ax2.set_title('Transport Gantt (Vehicle Tasks)')
    ax2.set_xlabel('Time')
    ax2.grid(axis='x', linestyle='-', alpha=0.1)

    # 底部公共图例
    handles = [plt.Rectangle((0,0),1,1, color=colors[k], label=k) for k in ['T1','T2','T3','T4']]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    plt.close(fig)

def plot_traditional_production(out_path):
    """还原第二张图：传统FIFO生产甘特图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    setup_style()
    colors = {'T1': '#3b71ed', 'T2': '#28a745', 'T3': '#f39c12', 'T4': '#ef5350'}
    
    # Production
    ax.broken_barh([(0, 34)], (0.6, 0.8), facecolors=colors['T1'], edgecolors='gray', linewidth=0.5)
    ax.text(17, 1, 'T1', color='white', fontweight='bold', ha='center', va='center', fontsize=8)
    
    ax.broken_barh([(37, 22)], (1.6, 0.8), facecolors=colors['T2'], edgecolors='gray', linewidth=0.5)
    ax.text(48, 2, 'T2', color='white', fontweight='bold', ha='center', va='center', fontsize=8)
    
    ax.broken_barh([(36, 40)], (0.6, 0.8), facecolors=colors['T3'], edgecolors='gray', linewidth=0.5)
    ax.text(56, 1, 'T3', color='white', fontweight='bold', ha='center', va='center', fontsize=8)
    
    ax.broken_barh([(76, 48)], (0.6, 0.8), facecolors=colors['T4'], edgecolors='gray', linewidth=0.5)
    ax.text(100, 1, 'T4', color='white', fontweight='bold', ha='center', va='center', fontsize=8)
    
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['y-005', 'y-002'])
    ax.set_ylabel('Machines')
    ax.set_xlabel('Time')
    ax.set_title('Traditional FIFO - Production Gantt', fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.1, color='lightblue')
    
    # 隐藏边框
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#e0e0e0'); ax.spines['bottom'].set_color('#e0e0e0')

    handles = [plt.Rectangle((0,0),1,1, color=colors[k], label=k) for k in ['T1','T2','T3','T4']]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    
    plt.tight_layout()
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    plt.close(fig)

def plot_traditional_transport(out_path):
    """还原第三张图：传统FIFO运输甘特图"""
    fig, ax = plt.subplots(figsize=(12, 5))
    setup_style()
    colors = {'T1': '#3b71ed', 'T2': '#28a745', 'T3': '#f39c12', 'T4': '#ef5350'}
    
    # Transport
    ax.broken_barh([(0, 2)], (1.6, 0.8), facecolors=colors['T1'], edgecolors='gray', linewidth=0.5)
    ax.text(1, 2, 'T1', color='white', fontweight='bold', ha='center', va='center', fontsize=8)
    
    ax.broken_barh([(34, 3)], (1.6, 0.8), facecolors=colors['T2'], edgecolors='gray', linewidth=0.5)
    ax.text(35.5, 2, 'T2', color='white', fontweight='bold', ha='center', va='center', fontsize=8)
    
    ax.broken_barh([(34, 2)], (0.6, 0.8), facecolors=colors['T3'], edgecolors='gray', linewidth=0.5)
    ax.text(35, 1, 'T3', color='white', fontweight='bold', ha='center', va='center', fontsize=8)
    
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['v2', 'v1'])
    ax.set_ylabel('Vehicles')
    ax.set_xlabel('Time')
    ax.set_title('Traditional FIFO - Transport Gantt', fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.1, color='lightblue')
    ax.set_xlim(0, 40)
    
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#e0e0e0'); ax.spines['bottom'].set_color('#e0e0e0')

    handles = [plt.Rectangle((0,0),1,1, color=colors[k], label=k) for k in ['T1','T2','T3','T4']]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
    
    plt.tight_layout()
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    setup_style()
    base_dir = Path(r"c:\Users\shiver\Desktop\Sched_LLM\backend\data\fjsp_epl_benchmark")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    plot_combined_gantt(base_dir / "recreated_gantt_combined_12pt.svg")
    plot_traditional_production(base_dir / "recreated_traditional_prod_12pt.svg")
    plot_traditional_transport(base_dir / "recreated_traditional_trans_12pt.svg")
    
    print(f"All Gantt plots recreated in 12pt and saved to {base_dir}")
