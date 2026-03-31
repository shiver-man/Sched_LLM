import argparse
import ast
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fjsp_case_runner import parse_fjs
from app.api.routes_simulation import _normalize_rich_payload_for_ppo
from app.models.schema import ScheduleRequest
from app.models.state import build_initial_state
from app.core.simulator import Simulator
from app.core.scheduler import PDR
from app.core.evaluator import Evaluator


def run_single_case(
    case_path: Path,
    max_steps: int,
    framework_rule: str = "COOP_RH",
    joint_weights: Dict | None = None,
) -> Dict:
    payload = parse_fjs(case_path)
    req_data = _normalize_rich_payload_for_ppo(payload)
    if joint_weights:
        metadata = req_data.get("metadata", {}) or {}
        dispatch_cfg = metadata.get("dispatching_config", {}) or {}
        dispatch_cfg["joint_score_weights"] = joint_weights
        metadata["dispatching_config"] = dispatch_cfg
        req_data["metadata"] = metadata
    req = ScheduleRequest(**req_data)
    base_state = build_initial_state(req)
    metrics_by_rule: Dict[str, Dict] = {}
    for rule in ["SPT", "FIFO", "MWKR", framework_rule]:
        state = copy.deepcopy(base_state)
        final_state = Simulator.run_simulation(
            state,
            lambda s, r=rule: PDR.get_dispatch_action(s, rule=r),
            max_steps=max_steps,
        )
        metrics_by_rule[rule] = Evaluator.evaluate(final_state)

    traditional_candidates = ["SPT", "FIFO", "MWKR"]
    complete_candidates = [r for r in traditional_candidates if metrics_by_rule[r].get("is_complete")]
    target_pool = complete_candidates if complete_candidates else traditional_candidates
    traditional_rule = min(target_pool, key=lambda r: metrics_by_rule[r]["makespan"])
    traditional_metrics = metrics_by_rule[traditional_rule]
    framework_metrics = metrics_by_rule[framework_rule]
    return {
        "case": case_path.stem,
        "traditional_rule": traditional_rule,
        "framework_rule": framework_rule,
        "traditional_metrics": traditional_metrics,
        "framework_metrics": framework_metrics,
    }


def plot_comparison(records: List[Dict], image_path: Path) -> Dict[str, float]:
    labels = [r["case"] for r in records]
    mk_tr = np.array([r["traditional_metrics"]["makespan"] for r in records], dtype=float)
    mk_fw = np.array([r["framework_metrics"]["makespan"] for r in records], dtype=float)
    tp_tr = np.array([r["traditional_metrics"]["total_transport_time"] for r in records], dtype=float)
    tp_fw = np.array([r["framework_metrics"]["total_transport_time"] for r in records], dtype=float)

    avg_mk_improve = float((mk_tr.mean() - mk_fw.mean()) / mk_tr.mean() * 100.0)
    avg_tp_improve = float((tp_tr.mean() - tp_fw.mean()) / tp_tr.mean() * 100.0)

    idx = np.arange(len(labels))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=150)

    axes[0].bar(idx - width / 2, mk_tr, width=width, label="Traditional (best of SPT/FIFO/MWKR)")
    framework_rule = records[0]["framework_rule"] if records else "Framework"
    axes[0].bar(idx + width / 2, mk_fw, width=width, label=f"Framework ({framework_rule})")
    axes[0].set_title("Makespan Comparison (lower is better)")
    axes[0].set_ylabel("Makespan")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(labels, rotation=30)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(idx - width / 2, tp_tr, width=width, label="Traditional (best of SPT/FIFO/MWKR)")
    axes[1].bar(idx + width / 2, tp_fw, width=width, label=f"Framework ({framework_rule})")
    axes[1].set_title("Total Transport Time Comparison (lower is better)")
    axes[1].set_ylabel("Total Transport Time")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(labels, rotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.suptitle(
        "Brandimarte Mk01-Mk10 Production-Transport Scheduling Comparison\n"
        f"Avg Makespan Improvement: {avg_mk_improve:.2f}% | Avg Transport-Time Improvement: {avg_tp_improve:.2f}%",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(image_path)
    return {"avg_makespan_improve_pct": avg_mk_improve, "avg_transport_improve_pct": avg_tp_improve}


def parse_weights_text(text: str) -> Dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items()}
    except Exception:
        pass
    cleaned = text.strip().strip("{}")
    if not cleaned:
        return None
    result = {}
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        result[key.strip().strip("'\"")] = float(value.strip().strip("'\""))
    return result or None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=r"c:\Users\shiver\Desktop\Sched_LLM\backend\FJSP_epl\FJSP算例\Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text",
    )
    parser.add_argument("--pattern", type=str, default="Mk*.fjs")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument(
        "--json-out",
        type=str,
        default=r"c:\Users\shiver\Desktop\Sched_LLM\backend\data\mk_brandimarte_comparison.json",
    )
    parser.add_argument(
        "--img-out",
        type=str,
        default=r"c:\Users\shiver\Desktop\Sched_LLM\backend\data\mk_brandimarte_comparison.png",
    )
    parser.add_argument("--weights-json", type=str, default="")
    parser.add_argument("--framework-rule", type=str, default="COOP_RH")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    case_files = sorted(dataset_dir.glob(args.pattern))
    if not case_files:
        raise ValueError(f"未找到任何算例文件: {dataset_dir} {args.pattern}")

    joint_weights = parse_weights_text(args.weights_json)
    framework_rule = str(args.framework_rule).strip().upper()
    records = [
        run_single_case(
            path,
            max_steps=args.max_steps,
            framework_rule=framework_rule,
            joint_weights=joint_weights,
        )
        for path in case_files
    ]

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    img_out = Path(args.img_out)
    img_out.parent.mkdir(parents=True, exist_ok=True)
    summary = plot_comparison(records, img_out)

    better_cases = sum(
        1
        for r in records
        if r["framework_metrics"]["makespan"] < r["traditional_metrics"]["makespan"]
    )

    print(f"cases: {len(records)}")
    print(f"framework_better_cases: {better_cases}/{len(records)}")
    print(f"avg_makespan_improve_pct: {summary['avg_makespan_improve_pct']:.4f}")
    print(f"avg_transport_improve_pct: {summary['avg_transport_improve_pct']:.4f}")
    print(f"json_saved_to: {json_out}")
    print(f"img_saved_to: {img_out}")
    print(f"framework_rule: {framework_rule}")
    if joint_weights:
        print(f"joint_weights: {json.dumps(joint_weights, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
