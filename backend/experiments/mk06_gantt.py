import json
import os
import sys
import copy
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.api.routes_simulation import _normalize_rich_payload_for_ppo
from app.core.evaluator import Evaluator
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


def _short_op_label(op_id: str) -> str:
    s = str(op_id or "")
    m = re.search(r"[Oo](\d+)$", s)
    if not m:
        m = re.search(r"(\d+)$", s)
    return f"T{m.group(1)}" if m else "T?"


def _show_label(start_time: float, finish_time: float, cutoff: float = 160.0) -> bool:
    return (finish_time - start_time) > 2.0 and start_time <= cutoff


def _best_traditional_rule(summary: List[Dict[str, Any]], rules: List[str]) -> Optional[str]:
    rule_set = {r.upper() for r in rules}
    traditional = [x for x in summary if str(x.get("rule", "")).upper() in rule_set]
    if not traditional:
        return None
    complete = [x for x in traditional if bool(x.get("is_complete", False))]
    pool = complete if complete else traditional
    best = min(pool, key=lambda x: float(x.get("makespan", 1e18)))
    return str(best.get("rule"))


def _plot_gantt(plan: List[Dict[str, Any]], title: str, out_file: Path) -> None:
    if not plan:
        raise RuntimeError(f"计划为空，无法绘图: {title}")

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
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, format="svg")
    plt.close(fig)


def _machine_end_times(plan: List[Dict[str, Any]]) -> Dict[str, float]:
    end_map: Dict[str, float] = {}
    for step in plan:
        m = str(step["machine_id"])
        end_map[m] = max(end_map.get(m, 0.0), float(step["finish_time"]))
    return end_map


def _pick_focus_window(framework_plan: List[Dict[str, Any]], traditional_plan: List[Dict[str, Any]]) -> tuple[float, float, str]:
    fw_end = _machine_end_times(framework_plan)
    tr_end = _machine_end_times(traditional_plan)
    all_machines = sorted(set(fw_end.keys()) | set(tr_end.keys()))
    worst_machine = max(all_machines, key=lambda m: fw_end.get(m, 0.0) - tr_end.get(m, 0.0))

    fw_map = {(str(s["job_id"]), str(s["op_id"])): s for s in framework_plan}
    tr_map = {(str(s["job_id"]), str(s["op_id"])): s for s in traditional_plan}
    common = sorted(set(fw_map.keys()) & set(tr_map.keys()), key=lambda k: float(fw_map[k]["start_time"]))

    anchor_start = 0.0
    anchor_finish = 0.0
    for key in common:
        fs = fw_map[key]
        ts = tr_map[key]
        if str(fs["machine_id"]) != worst_machine:
            continue
        if float(fs["finish_time"]) - float(ts["finish_time"]) > 1e-6:
            anchor_start = min(float(fs["start_time"]), float(ts["start_time"]))
            anchor_finish = max(float(fs["finish_time"]), float(ts["finish_time"]))
            break

    if anchor_finish <= anchor_start:
        anchor_finish = max(fw_end.get(worst_machine, 0.0), tr_end.get(worst_machine, 0.0))
        anchor_start = max(0.0, anchor_finish - 80.0)

    # 关键片段窗口：前后各扩展 25 时间单位
    t0 = max(0.0, anchor_start - 25.0)
    t1 = anchor_finish + 25.0
    return t0, t1, worst_machine


def _plot_gantt_focus(
    framework_plan: List[Dict[str, Any]],
    traditional_plan: List[Dict[str, Any]],
    t0: float,
    t1: float,
    focus_machine: str,
    out_file: Path,
    trad_rule: str,
) -> None:
    def _filter(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for s in plan:
            st = float(s["start_time"])
            ft = float(s["finish_time"])
            if ft < t0 or st > t1:
                continue
            out.append(s)
        return out

    fw = _filter(framework_plan)
    tr = _filter(traditional_plan)
    machines = sorted({str(s["machine_id"]) for s in fw + tr})
    m_index = {m: i for i, m in enumerate(machines)}
    jobs = sorted({str(s["job_id"]) for s in fw + tr})
    cmap = plt.get_cmap("tab20")
    job_color = {j: cmap(i % 20) for i, j in enumerate(jobs)}

    fig, axes = plt.subplots(2, 1, figsize=(14, max(7, 0.9 * len(machines))), sharex=True)
    plans = [("Framework (COOP_RH)", fw), (f"Traditional ({trad_rule})", tr)]
    for ax, (title, plan) in zip(axes, plans):
        for s in plan:
            m = str(s["machine_id"])
            y = m_index[m]
            st = float(s["start_time"])
            ft = float(s["finish_time"])
            w = max(0.0, ft - st)
            op = str(s.get("op_id", ""))
            job = str(s["job_id"])
            edge = "red" if m == focus_machine else "black"
            lw = 1.0 if m == focus_machine else 0.4
            ax.broken_barh([(st, w)], (y - 0.35, 0.7), facecolors=job_color[job], edgecolors=edge, linewidth=lw)
            if _show_label(st, ft):
                ax.text(st + w / 2.0, y, _short_op_label(op), ha="center", va="center", fontsize=7)
        ax.set_yticks(range(len(machines)))
        ax.set_yticklabels(machines)
        ax.set_ylabel("Machine")
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
    axes[-1].set_xlabel("Time")
    axes[-1].set_xlim(t0, t1)
    fig.suptitle(f"MK06 Key Segment Gantt (focus machine: {focus_machine}, window: {t0:.1f}-{t1:.1f})")
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, format="svg")
    plt.close(fig)


def _run_rule(initial_state: Dict[str, Any], rule: str, max_steps: int = 1200) -> Dict[str, Any]:
    state_copy = copy.deepcopy(initial_state)
    final_state = Simulator.run_simulation(
        state_copy,
        lambda s: PDR.get_dispatch_action(s, rule=rule),
        max_steps=max_steps,
    )
    metrics = Evaluator.evaluate(final_state)
    return {
        "rule": rule,
        "metrics": metrics,
        "plan": final_state.get("history", []),
    }


def main() -> None:
    _set_font()
    mk06_path = Path(
        r"c:\Users\shiver\Desktop\Sched_LLM\backend\FJSP_epl\FJSP算例\Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk06.fjs"
    )
    payload = parse_fjs(mk06_path)
    req_data = _normalize_rich_payload_for_ppo(payload)
    req = ScheduleRequest(**req_data)
    initial_state = build_initial_state(req)

    framework_rule = "COOP_RH"
    traditional_rules = ["SPT", "FIFO", "MWKR"]
    all_rules = traditional_rules + [framework_rule]
    results = [_run_rule(initial_state, rule) for rule in all_rules]

    summary = [
        {
            "rule": r["rule"],
            "makespan": r["metrics"]["makespan"],
            "is_complete": r["metrics"]["is_complete"],
            "num_events": r["metrics"]["num_events"],
        }
        for r in results
    ]
    result_map = {r["rule"].upper(): r for r in results}

    framework_scheme = result_map.get(framework_rule.upper())
    if not framework_scheme:
        raise RuntimeError("未找到 COOP_RH 的方案。")

    best_tr_rule = _best_traditional_rule(summary, traditional_rules)
    if not best_tr_rule:
        raise RuntimeError("未找到传统规则方案（SPT/FIFO/MWKR）。")
    traditional_scheme = result_map.get(best_tr_rule.upper())
    if not traditional_scheme:
        raise RuntimeError(f"未找到传统最优方案的详细计划: {best_tr_rule}")

    out_dir = Path(r"c:\Users\shiver\Desktop\Sched_LLM\backend\data\fjsp_epl_benchmark")
    fw_file = out_dir / "mk06_gantt_framework_coop_rh.svg"
    tr_file = out_dir / f"mk06_gantt_traditional_{best_tr_rule.lower()}.svg"

    _plot_gantt(
        framework_scheme.get("plan") or [],
        "MK06 Gantt - Framework (COOP_RH)",
        fw_file,
    )
    _plot_gantt(
        traditional_scheme.get("plan") or [],
        f"MK06 Gantt - Traditional ({best_tr_rule})",
        tr_file,
    )

    fw_plan = framework_scheme.get("plan") or []
    tr_plan = traditional_scheme.get("plan") or []
    t0, t1, focus_machine = _pick_focus_window(fw_plan, tr_plan)
    key_seg_file = out_dir / "mk06_gantt_key_segment_framework_vs_traditional.svg"
    _plot_gantt_focus(
        fw_plan,
        tr_plan,
        t0=t0,
        t1=t1,
        focus_machine=focus_machine,
        out_file=key_seg_file,
        trad_rule=best_tr_rule,
    )

    print(
        json.dumps(
            {
                "framework_rule": framework_rule,
                "traditional_best_rule": best_tr_rule,
                "framework_svg": str(fw_file),
                "traditional_svg": str(tr_file),
                "key_segment_svg": str(key_seg_file),
                "focus_machine": focus_machine,
                "focus_window": [round(t0, 3), round(t1, 3)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

