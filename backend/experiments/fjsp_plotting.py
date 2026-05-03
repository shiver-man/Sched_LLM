import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import re

FIFTH_FONT_SIZE = 12.0


def _set_font():
    plt.rcParams.update(
        {
            "font.size": FIFTH_FONT_SIZE,
            "axes.titlesize": FIFTH_FONT_SIZE,
            "axes.labelsize": FIFTH_FONT_SIZE,
            "xtick.labelsize": FIFTH_FONT_SIZE,
            "ytick.labelsize": FIFTH_FONT_SIZE,
            "legend.fontsize": FIFTH_FONT_SIZE,
            "figure.titlesize": FIFTH_FONT_SIZE,
        }
    )

def _machine_wait_col(df: pd.DataFrame, framework: bool) -> str:
    # 优先取已聚合好的 sync_loss 字段；若没有，再回退到 waiting_* 列组合
    direct = "framework_sync_loss" if framework else "traditional_sync_loss"
    if direct in df.columns:
        return direct
    return direct

def plot_fjsp_results():
    _set_font()
    base_dir = Path(r"c:\Users\shiver\Desktop\Sched_LLM\backend\data\fjsp_epl_benchmark")
    main_json = base_dir / "main_makespan_results.json"
    ext_json = base_dir / "extended_waiting_results.json"
    
    if not main_json.exists() or not ext_json.exists():
        print("Results files not found.")
        return

    with open(main_json, "r", encoding="utf-8") as f:
        main_data = json.load(f)
    with open(ext_json, "r", encoding="utf-8") as f:
        ext_data = json.load(f)

    df_main = pd.DataFrame(main_data)
    df_ext = pd.DataFrame(ext_data)

    # 仅保留 Brandimarte 的 Mk01-Mk10
    mk_pattern = re.compile(r"^Mk\d{2}$", re.IGNORECASE)
    df_main = df_main[df_main["group"].astype(str).str.lower() == "brandimarte"].copy()
    df_ext = df_ext[df_ext["group"].astype(str).str.lower() == "brandimarte"].copy()
    df_main = df_main[df_main["case"].astype(str).str.match(mk_pattern)].copy()
    df_ext = df_ext[df_ext["case"].astype(str).str.match(mk_pattern)].copy()
    df_main = df_main.sort_values("case")
    df_ext = df_ext.sort_values("case")
    
    # Filter valid data
    df_main = df_main.dropna(subset=["framework_makespan", "traditional_makespan"])
    
    # 1. Bar Chart: Makespan Comparison (Sample 10 cases)
    sample_df = df_main.head(10)
    plt.figure(figsize=(12, 6))
    x = range(len(sample_df))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], sample_df["traditional_makespan"], width, label='Traditional (Best)', color='gray', alpha=0.7)
    plt.bar([i + width/2 for i in x], sample_df["framework_makespan"], width, label='Framework (LLM)', color='skyblue')
    
    plt.xlabel('Case')
    plt.ylabel('Makespan')
    plt.title('Makespan Comparison: Traditional vs Framework (Sample Cases)')
    plt.xticks(x, sample_df["case"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "brandimarte_mk01_mk10_makespan_comparison_bar.svg", format='svg')
    plt.close()

    # 2. Bar Chart: Sync Loss Comparison (Sample 10 cases)
    sample_ext = df_ext.head(10)
    plt.figure(figsize=(12, 6))
    
    plt.bar([i - width/2 for i in x], sample_ext["traditional_sync_loss"], width, label='Traditional Sync Loss', color='lightcoral', alpha=0.7)
    plt.bar([i + width/2 for i in x], sample_ext["framework_sync_loss"], width, label='Framework Sync Loss', color='lightgreen')
    
    plt.xlabel('Case')
    plt.ylabel('Sync Loss (Waiting Time)')
    plt.title('Synchronization Loss Comparison (Sample Cases)')
    plt.xticks(x, sample_ext["case"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "sync_loss_comparison_bar.svg", format='svg')
    plt.close()

    # 3. Bar Chart: Machine Waiting Time Comparison (Sample 10 cases)
    sample_wait = df_ext.head(10)
    plt.figure(figsize=(12, 6))
    fw_wait_col = _machine_wait_col(sample_wait, framework=True)
    tr_wait_col = _machine_wait_col(sample_wait, framework=False)
    plt.bar([i - width/2 for i in x], sample_wait[tr_wait_col], width, label='Traditional Machine Waiting', color='navajowhite', alpha=0.8)
    plt.bar([i + width/2 for i in x], sample_wait[fw_wait_col], width, label='Framework Machine Waiting', color='mediumseagreen', alpha=0.9)
    plt.xlabel('Case')
    plt.ylabel('Machine Waiting Time')
    plt.title('Machine Waiting Time Comparison (Sample Cases)')
    plt.xticks(x, sample_wait["case"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "brandimarte_mk01_mk10_machine_waiting_comparison_bar.svg", format='svg')
    plt.close()

    # 4. 协同等待代理图（以 sync_loss 作为代理）
    proxy_df = pd.merge(
        df_main[["case"]],
        df_ext[["case", "traditional_sync_loss", "framework_sync_loss"]],
        on="case",
        how="inner",
    ).sort_values("case")

    plt.figure(figsize=(12, 6))
    px = range(len(proxy_df))
    pwidth = 0.35
    plt.bar(
        [i - pwidth / 2 for i in px],
        proxy_df["traditional_sync_loss"],
        pwidth,
        label="Traditional Sync Proxy",
        color="salmon",
        alpha=0.8,
    )
    plt.bar(
        [i + pwidth / 2 for i in px],
        proxy_df["framework_sync_loss"],
        pwidth,
        label="Framework Sync Proxy",
        color="seagreen",
        alpha=0.85,
    )
    plt.xlabel("Case")
    plt.ylabel("Collaborative Waiting Proxy")
    plt.title("Brandimarte Mk01-Mk10 Collaborative Waiting Proxy")
    plt.xticks(px, proxy_df["case"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "brandimarte_mk01_mk10_collab_wait_proxy_bar.svg", format='svg')
    plt.close()

    # 5. Framework-only boxplot: makespan + machine waiting
    framework_mk = df_main["framework_makespan"].dropna()
    framework_wait = df_ext["framework_sync_loss"].dropna() if "framework_sync_loss" in df_ext.columns else pd.Series(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].boxplot([framework_mk], tick_labels=["Framework"])
    axes[0].set_title("Framework Makespan")
    axes[0].set_ylabel("Makespan")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].boxplot([framework_wait], tick_labels=["Framework"])
    axes[1].set_title("Framework Machine Waiting")
    axes[1].set_ylabel("Machine Waiting Time")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Framework Boxplot: Makespan and Machine Waiting")
    plt.tight_layout()
    plt.savefig(base_dir / "framework_makespan_machine_wait_boxplot.svg", format='svg')
    plt.close()

    print(f"Plots saved as SVG in {base_dir}")

if __name__ == "__main__":
    plot_fjsp_results()
