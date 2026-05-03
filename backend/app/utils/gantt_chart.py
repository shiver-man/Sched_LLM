import io
import re
import base64
from typing import List, Dict, Any
import matplotlib
import matplotlib.pyplot as plt

# Use Agg backend for non-interactive image generation (safe for web servers)
matplotlib.use("Agg")

def _short_op_label(op_id: str) -> str:
    s = str(op_id or "")
    m = re.search(r"[Oo](\d+)$", s)
    if not m:
        m = re.search(r"(\d+)$", s)
    return f"T{m.group(1)}" if m else "T?"

def _show_label(start_time: float, finish_time: float, cutoff: float = float('inf')) -> bool:
    return (finish_time - start_time) > 2.0 and start_time <= cutoff

def generate_gantt_base64(plan: List[Dict[str, Any]], title: str = "Schedule Gantt Chart") -> str:
    """
    Generate a Gantt chart SVG and return it as a Base64 encoded string.
    """
    if not plan:
        return ""

    plt.rcParams.update({
        "font.size": 12.0,
        "axes.titlesize": 12.0,
        "axes.labelsize": 12.0,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,
        "legend.fontsize": 12.0,
        "figure.titlesize": 12.0,
    })

    machines = sorted({str(step["machine_id"]) for step in plan})
    m_index = {m: i for i, m in enumerate(machines)}
    jobs = sorted({str(step["job_id"]) for step in plan})
    cmap = plt.get_cmap("tab20")
    job_color = {j: cmap(i % 20) for i, j in enumerate(jobs)}

    fig, ax = plt.subplots(figsize=(14, max(5, 0.7 * len(machines))))
    for step in plan:
        m = str(step["machine_id"])
        y = m_index[m]
        start = float(step["start_time"])
        finish = float(step["finish_time"])
        width = max(0.0, finish - start)
        op = str(step.get("op_id", ""))
        job = str(step["job_id"])
        
        ax.broken_barh([(start, width)], (y - 0.35, 0.7), facecolors=job_color[job], edgecolors="black", linewidth=0.4)
        if _show_label(start, finish):
            ax.text(start + width / 2.0, y, _short_op_label(op), ha="center", va="center", fontsize=7)

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    
    # Encode the SVG bytes to base64
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    
    # Return as a data URL
    return f"data:image/svg+xml;base64,{b64_str}"
