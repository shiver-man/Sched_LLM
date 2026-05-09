from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_PATH = Path(r"c:\Users\shiver\Desktop\Sched_LLM\backend\data\research_framework_sample_style.svg")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "font.size": 11,
        }
    )


def draw_box(ax, x, y, w, h, text, fontsize=11, dashed=False, bold=False, fc="white"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1,
        edgecolor="#666666",
        facecolor=fc,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        wrap=True,
    )


def draw_group(ax, x, y, w, h, title):
    rect = Rectangle(
        (x, y),
        w,
        h,
        linewidth=1.0,
        edgecolor="#9a9a9a",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)
    draw_box(ax, x + 0.2, y + h - 0.55, 1.6, 0.38, title, fontsize=10.5, bold=True, fc="#f7f7f7")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#666666",
        )
    )


def main():
    setup_style()
    fig, ax = plt.subplots(figsize=(16, 10.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Outer frame
    ax.add_patch(Rectangle((0.15, 0.15), 15.7, 9.7, linewidth=1.2, edgecolor="#777777", facecolor="none"))

    # Left guide column
    left_x = 0.55
    labels = [
        ("现实问题", 9.0),
        ("难点/挑战", 8.1),
        ("关键问题", 7.2),
        ("核心思路", 6.1),
        ("研究内容", 5.0),
    ]
    for text, y in labels:
        draw_box(ax, left_x, y, 1.35, 0.42, text, fontsize=10.5, bold=True, fc="#f4f4f4")
    for idx in range(len(labels) - 1):
        arrow(ax, left_x + 0.68, labels[idx][1], left_x + 0.68, labels[idx + 1][1] + 0.42)

    # Top problem summary
    draw_box(
        ax,
        2.2,
        9.0,
        13.0,
        0.52,
        "多源不确定扰动下生产-运输一体化柔性作业车间的协同优化决策与自适应控制",
        fontsize=13,
        bold=True,
        fc="#cfe2f3",
    )

    draw_box(ax, 2.6, 8.2, 4.0, 0.52, "生产资源耦合强，调度空间离散且复杂", fc="#efe6bf")
    draw_box(ax, 6.95, 8.2, 3.8, 0.52, "多源不确定性对制造过程影响大", fc="#d9ead3")
    draw_box(ax, 11.1, 8.2, 3.7, 0.52, "扰动下需实现动态决策更新与自适应控制", fc="#f4cccc")
    arrow(ax, 4.6, 8.2, 4.6, 7.85)
    arrow(ax, 8.85, 8.2, 8.85, 7.85)
    arrow(ax, 12.95, 8.2, 12.95, 7.85)

    draw_box(
        ax,
        3.8,
        7.2,
        9.5,
        0.78,
        "1. 如何分析生产资源间的约束限制、交互关系？\n"
        "2. 如何在多源不确定性扰动环境下进行实时动态决策更新和执行控制？",
        fontsize=11.2,
        bold=True,
        fc="#f7f7f7",
    )

    draw_box(
        ax,
        2.2,
        6.2,
        13.0,
        0.56,
        "结合“统一状态建模 + 多算法优化求解 + 大模型可解释分析”，构建生产与物流同步决策框架",
        fontsize=11.8,
        bold=True,
        fc="#f7f7f7",
    )

    # Bottom research region
    draw_group(ax, 2.1, 0.95, 13.15, 4.95, "研究内容")

    # Left vertical bar
    draw_box(ax, 2.3, 1.35, 0.42, 4.1, "基于生产-运输协同优化的智能调度框架", fontsize=11, bold=True, fc="#4a78c9")
    ax.text(2.51, 3.4, "基于生产-运输协同优化的智能调度框架", rotation=90, ha="center", va="center", fontsize=11, color="white", fontweight="bold")

    # Module 1
    draw_group(ax, 2.8, 1.1, 4.25, 4.5, "研究内容1")
    draw_box(ax, 3.2, 5.1, 3.45, 0.38, "柔性作业车间生产-运输协同调度建模方法", fontsize=10.4, fc="#d9ead3")
    draw_box(ax, 3.05, 4.05, 1.65, 0.95, "建模对象\n工序\n机器\n车辆\n布局", fontsize=10.2)
    draw_box(ax, 4.95, 4.05, 1.8, 0.95, "关键约束\n工序顺序\n资源冲突\n运输可达性\n时间窗", fontsize=10.2)
    ax.text(4.83, 4.5, "映射", ha="center", va="center", fontsize=10, color="#666666")
    arrow(ax, 4.7, 4.5, 4.95, 4.5)
    draw_box(ax, 3.05, 2.88, 3.7, 0.72, "统一状态表达\njobs + machines + vehicles + layout + metadata", fontsize=10.3)
    draw_box(ax, 3.05, 1.85, 3.7, 0.72, "状态特征提取\n当前时间、机器队列、在制品位置、运输资源状态", fontsize=10.3)
    draw_box(ax, 3.3, 1.22, 3.2, 0.42, "输出：统一状态空间与约束表示", fontsize=10.3, bold=True, fc="#f7f7f7")
    arrow(ax, 4.9, 4.05, 4.9, 3.62)
    arrow(ax, 4.9, 2.88, 4.9, 2.58)
    arrow(ax, 4.9, 1.85, 4.9, 1.64)

    # Module 2
    draw_group(ax, 7.25, 1.1, 3.85, 4.5, "研究内容2")
    draw_box(ax, 7.55, 5.1, 3.25, 0.38, "面向多维度协同目标的多算法优化求解方法", fontsize=10.3, fc="#fff2cc")
    draw_box(ax, 7.55, 4.05, 3.25, 0.82, "初始解生成\nSPT / MWKR / FIFO 等规则快速构造候选解", fontsize=10.2)
    draw_box(ax, 7.55, 3.0, 3.25, 0.72, "协同优化\n元启发式全局搜索 + 局部搜索精细改进", fontsize=10.2)
    draw_box(ax, 7.55, 2.0, 3.25, 0.72, "评价指标\nMakespan、机器等待、运输等待、协同损失", fontsize=10.2)
    draw_box(ax, 7.8, 1.22, 2.75, 0.42, "输出：候选方案与最优调度方案", fontsize=10.3, bold=True, fc="#f7f7f7")
    arrow(ax, 9.18, 4.05, 9.18, 3.72)
    arrow(ax, 9.18, 3.0, 9.18, 2.72)
    arrow(ax, 9.18, 2.0, 9.18, 1.64)

    # Module 3
    draw_group(ax, 11.25, 1.1, 3.75, 4.5, "研究内容3")
    draw_box(ax, 11.5, 5.1, 3.25, 0.38, "基于大语言模型的方案分析与动态反馈方法", fontsize=10.2, fc="#fce5cd")
    draw_box(ax, 11.5, 4.0, 3.25, 0.88, "输入\n最优方案、对比指标、关键轨迹、瓶颈设备与等待片段", fontsize=10.1)
    draw_box(ax, 11.5, 2.95, 3.25, 0.78, "分析解释\n为什么更优、瓶颈在哪里、协同等待为何升降", fontsize=10.1)
    draw_box(ax, 11.5, 2.05, 3.25, 0.62, "经验反馈\n规则适用场景、扰动下调度调整建议", fontsize=10.1)
    draw_box(ax, 11.95, 1.52, 2.35, 0.32, "Qwen2 / Llama", fontsize=10.0, fc="#fafafa")
    draw_box(ax, 11.7, 1.12, 2.85, 0.28, "输出：可解释报告与策略知识", fontsize=9.9, bold=True, fc="#f7f7f7")
    arrow(ax, 13.13, 4.0, 13.13, 3.73)
    arrow(ax, 13.13, 2.95, 13.13, 2.67)
    arrow(ax, 13.13, 2.05, 13.13, 1.84)

    # Cross-module linkage
    ax.text(7.1, 3.35, "状态输入", ha="center", va="center", fontsize=10, color="#666666")
    arrow(ax, 7.05, 3.35, 7.25, 3.35)
    ax.text(11.15, 3.35, "方案与指标输入", ha="center", va="center", fontsize=10, color="#666666")
    arrow(ax, 11.1, 3.35, 11.25, 3.35)

    # Bottom summary
    draw_box(ax, 4.3, 0.35, 8.9, 0.42, "面向动态环境的智能调度决策框架", fontsize=15, bold=True, fc="#f7f7f7")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(OUT_PATH, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
