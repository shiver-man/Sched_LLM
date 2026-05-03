import copy
import time
import os
import sys
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.api.routes_simulation import _normalize_rich_payload_for_ppo
from app.core.scheduler import PDR
from app.core.simulator import Simulator
from app.models.schema import ScheduleRequest
from app.models.state import build_initial_state
from experiments.fjsp_case_runner import parse_fjs

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


def measure_runtime():
    base = Path(
        r"c:\Users\shiver\Desktop\Sched_LLM\backend\FJSP_epl\FJSP算例\Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text"
    )
    files = [base / f"Mk{str(i).zfill(2)}.fjs" for i in range(1, 11)]
    rules = ["SPT", "FIFO", "MWKR", "COOP_RH"]

    rows = []
    for f in files:
        payload = parse_fjs(f)
        req = ScheduleRequest(**_normalize_rich_payload_for_ppo(payload))
        init_state = build_initial_state(req)
        row = {"case": f.stem}
        for rule in rules:
            t0 = time.perf_counter()
            Simulator.run_simulation(
                copy.deepcopy(init_state),
                lambda s, rr=rule: PDR.get_dispatch_action(s, rule=rr),
                max_steps=1200,
            )
            row[rule] = time.perf_counter() - t0
        rows.append(row)
    return rows, rules


def plot_runtime(rows, rules):
    out_dir = Path(r"c:\Users\shiver\Desktop\Sched_LLM\backend\data\fjsp_epl_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [r["case"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：逐算例运行时间
    x = list(range(len(cases)))
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    colors = {
        "SPT": "#4C78A8",
        "FIFO": "#F58518",
        "MWKR": "#54A24B",
        "COOP_RH": "#E45756",
    }
    for i, rule in enumerate(rules):
        y = [r[rule] for r in rows]
        axes[0].bar([v + offsets[i] for v in x], y, width=width, label=rule, color=colors[rule], alpha=0.9)
    axes[0].set_title("MK01-MK10 Runtime Per Case")
    axes[0].set_xlabel("Case")
    axes[0].set_ylabel("Runtime (seconds)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cases, rotation=45)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    # 右图：平均运行时间
    means = [mean([r[rule] for r in rows]) for rule in rules]
    axes[1].bar(rules, means, color=[colors[r] for r in rules], alpha=0.9)
    axes[1].set_title("Average Runtime (MK01-MK10)")
    axes[1].set_ylabel("Runtime (seconds)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    for i, v in enumerate(means):
        axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_file = out_dir / "brandimarte_mk01_mk10_runtime_comparison.svg"
    fig.savefig(out_file, format="svg")
    plt.close(fig)
    return out_file


def main():
    _set_font()
    rows, rules = measure_runtime()
    out_file = plot_runtime(rows, rules)
    print(f"saved: {out_file}")


if __name__ == "__main__":
    main()

