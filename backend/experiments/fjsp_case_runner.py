import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import sys

from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app


def parse_fjs(file_path: Path) -> Dict[str, Any]:
    raw_lines = [line.strip() for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    first = raw_lines[0].split()
    job_count = int(first[0])
    machine_count = int(first[1])
    jobs: List[Dict[str, Any]] = []
    for j in range(job_count):
        values = list(map(int, raw_lines[j + 1].split()))
        p = 0
        op_count = values[p]
        p += 1
        operations = []
        for o in range(op_count):
            k = values[p]
            p += 1
            cms = []
            for _ in range(k):
                m = values[p]
                t = values[p + 1]
                p += 2
                cms.append({"machine_id": f"M{m}", "processing_time": float(t)})
            operations.append({"operation_id": f"J{j + 1}O{o + 1}", "candidate_machines": cms})
        jobs.append(
            {
                "job_id": f"J{j + 1}",
                "release_time": 0.0,
                "due_time": 100000.0,
                "initial_location": "L/U",
                "operations": operations,
            }
        )
    machines = [
        {"machine_id": f"M{i + 1}", "type": "FJSP_MACHINE", "location": f"L{i + 1}", "status": "idle"}
        for i in range(machine_count)
    ]
    nodes = ["L/U"] + [f"L{i + 1}" for i in range(machine_count)]
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            edges.append({"from": nodes[i], "to": nodes[j], "distance": float(abs(i - j) + 1)})
    vehicles = [
        {"vehicle_id": "V1", "start_location": "L/U", "speed": 1.0, "load_unload_time": 1.0, "status": "idle"},
        {"vehicle_id": "V2", "start_location": "L/U", "speed": 1.2, "load_unload_time": 1.0, "status": "idle"},
    ]
    return {
        "factory_info": {"factory_id": file_path.stem, "factory_name": "FJSP Bench", "planning_horizon": 50000.0, "current_time": 0.0},
        "shop_floor": {"machines": machines, "vehicles": vehicles, "transport_network": {"nodes": nodes, "edges": edges}},
        "jobs": jobs,
        "simulation_config": {"random_seed": 42, "ppo_max_steps": 800},
        "dispatching_config": {"ppo_policy_id": "latest", "transport_rule": "LOAD_BALANCING"},
        "llm_config": {"use_ollama": False},
        "return_raw_json": True,
    }


def _pick_by_rule(summary: List[Dict[str, Any]], rule: str) -> Dict[str, Any] | None:
    return next((x for x in summary if str(x.get("rule", "")).upper() == rule.upper()), None)


def _best_traditional(summary: List[Dict[str, Any]], traditional_rules: List[str]) -> Dict[str, Any] | None:
    candidates = [x for x in summary if str(x.get("rule", "")).upper() in {r.upper() for r in traditional_rules}]
    if not candidates:
        return None
    complete = [x for x in candidates if bool(x.get("is_complete", False))]
    pool = complete if complete else candidates
    return min(pool, key=lambda x: float(x.get("makespan", 1e18)))


def _best_by_rules(summary: List[Dict[str, Any]], rules: List[str]) -> Optional[Dict[str, Any]]:
    rule_set = {r.upper() for r in rules}
    candidates = [x for x in summary if str(x.get("rule", "")).upper() in rule_set]
    if not candidates:
        return None
    complete = [x for x in candidates if bool(x.get("is_complete", False))]
    pool = complete if complete else candidates
    return min(pool, key=lambda x: float(x.get("makespan", 1e18)))


def _extract_sync_loss(entry: Dict[str, Any]) -> float:
    idle_durations = (entry or {}).get("machine_idle_reason_durations", {}) or {}
    return float(idle_durations.get("waiting_transport", 0.0)) + float(idle_durations.get("waiting_machine", 0.0))


def run_cases(
    case_files: List[Path],
    framework_rule: str,
    traditional_rules: List[str],
    portfolio_rules: List[str],
    call_run_alias: bool = False,
) -> List[Dict[str, Any]]:
    client = TestClient(app)
    results = []
    framework_rule_u = framework_rule.upper()
    # 只运行必要策略，且允许组合框架
    selected_strategies = list(dict.fromkeys(traditional_rules + portfolio_rules + [framework_rule_u]))
    
    for idx, file_path in enumerate(case_files, start=1):
        payload = parse_fjs(file_path)
        payload["selected_strategies"] = selected_strategies
        entry: Dict[str, Any] = {"case": str(file_path), "http_status": None}
        if call_run_alias:
            resp = client.post("/run", json=payload)
            entry["http_status"] = resp.status_code
            if resp.status_code != 200:
                entry["error"] = resp.text
                results.append(entry)
                print(f"[case_done] {idx}/{len(case_files)} {file_path.name} status={resp.status_code}", flush=True)
                continue
        full = client.post("/simulation/run", json=payload)
        entry["http_status_full"] = full.status_code
        if full.status_code != 200:
            entry["error"] = full.text
            results.append(entry)
            continue
        body = full.json()
        summary = body.get("summary_comparison") or []
        schemes = body.get("detailed_schemes") or []
        entry["best_rule"] = summary[0]["rule"] if summary else None
        entry["best_makespan"] = summary[0]["makespan"] if summary else None
        entry["num_schemes"] = len(schemes)
        ga = next((x for x in summary if x.get("rule") == "GA"), None)
        entry["ga_makespan"] = ga.get("makespan") if ga else None
        entry["ga_complete"] = ga.get("is_complete") if ga else None
        entry["llm_brief_len"] = len(body.get("llm_readable_brief", ""))
        if framework_rule_u == "PORTFOLIO":
            fw = _best_by_rules(summary, portfolio_rules + traditional_rules)
            if fw:
                # 保留框架的真实来源策略，方便论文追踪
                fw = dict(fw)
                fw["rule"] = f"PORTFOLIO<{fw.get('rule')}>"
        else:
            fw = _pick_by_rule(summary, framework_rule_u)
        tr = _best_traditional(summary, traditional_rules)
        entry["framework_rule"] = framework_rule_u
        entry["framework_metrics"] = fw
        entry["traditional_best_metrics"] = tr
        if fw and tr:
            tr_mk = float(tr.get("makespan", 0.0))
            fw_mk = float(fw.get("makespan", 0.0))
            entry["makespan_improve_pct"] = ((tr_mk - fw_mk) / tr_mk * 100.0) if tr_mk > 0 else None
        else:
            entry["makespan_improve_pct"] = None
        results.append(entry)
        print(f"[case_done] {idx}/{len(case_files)} {file_path.name} status={entry.get('http_status_full')}", flush=True)
    return results


def _summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [x for x in rows if x.get("framework_metrics") and x.get("traditional_best_metrics")]
    if not valid:
        return {
            "cases": len(rows),
            "valid_cases": 0,
            "framework_win_cases": 0,
            "avg_makespan_improve_pct": None,
            "avg_transport_improve_pct": None,
            "avg_sync_loss_improve_pct": None,
        }

    fw_win = 0
    mk_improves = []
    tp_improves = []
    sync_improves = []
    for r in valid:
        fw = r["framework_metrics"]
        tr = r["traditional_best_metrics"]
        fw_mk = float(fw.get("makespan", 0.0))
        tr_mk = float(tr.get("makespan", 0.0))
        if fw_mk < tr_mk:
            fw_win += 1
        if tr_mk > 0:
            mk_improves.append((tr_mk - fw_mk) / tr_mk * 100.0)

        fw_tp = float(fw.get("transport_time", 0.0))
        tr_tp = float(tr.get("transport_time", 0.0))
        if tr_tp > 0:
            tp_improves.append((tr_tp - fw_tp) / tr_tp * 100.0)

        fw_sync = _extract_sync_loss(fw)
        tr_sync = _extract_sync_loss(tr)
        if tr_sync > 0:
            sync_improves.append((tr_sync - fw_sync) / tr_sync * 100.0)

    return {
        "cases": len(rows),
        "valid_cases": len(valid),
        "framework_win_cases": fw_win,
        "avg_makespan_improve_pct": round(sum(mk_improves) / len(mk_improves), 4) if mk_improves else None,
        "avg_transport_improve_pct": round(sum(tp_improves) / len(tp_improves), 4) if tp_improves else None,
        "avg_sync_loss_improve_pct": round(sum(sync_improves) / len(sync_improves), 4) if sync_improves else None,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _expand_main_rows(group_name: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        fw = r.get("framework_metrics") or {}
        tr = r.get("traditional_best_metrics") or {}
        out.append(
            {
                "group": group_name,
                "case": Path(r["case"]).stem,
                "framework_rule": r.get("framework_rule"),
                "framework_makespan": fw.get("makespan"),
                "traditional_rule": tr.get("rule"),
                "traditional_makespan": tr.get("makespan"),
                "makespan_improve_pct": r.get("makespan_improve_pct"),
                "framework_is_complete": fw.get("is_complete"),
                "traditional_is_complete": tr.get("is_complete"),
                "http_status_full": r.get("http_status_full"),
            }
        )
    return out


def _expand_extended_rows(group_name: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        fw = r.get("framework_metrics") or {}
        tr = r.get("traditional_best_metrics") or {}
        out.append(
            {
                "group": group_name,
                "case": Path(r["case"]).stem,
                "framework_transport_time": fw.get("transport_time"),
                "traditional_transport_time": tr.get("transport_time"),
                "framework_transport_wait": fw.get("transport_wait_time"),
                "traditional_transport_wait": tr.get("transport_wait_time"),
                "framework_sync_loss": _extract_sync_loss(fw),
                "traditional_sync_loss": _extract_sync_loss(tr),
                "busiest_vehicle_fw": fw.get("busiest_vehicle"),
                "busiest_vehicle_tr": tr.get("busiest_vehicle"),
            }
        )
    return out


def _split_batches(items: List[Path], batch_size: int) -> List[List[Path]]:
    if batch_size <= 0:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def collect_case_groups(dataset_root: Path) -> Dict[str, List[Path]]:
    return {
        "Brandimarte": sorted((dataset_root / "Brandimarte_Data" / "Text").glob("Mk*.fjs")),
        "Dauzere": sorted((dataset_root / "Dauzere_Data" / "Text").glob("*.fjs")),
        "Hurink_edata": sorted((dataset_root / "Hurink_Data" / "Text" / "edata").glob("*.fjs")),
        "Hurink_rdata": sorted((dataset_root / "Hurink_Data" / "Text" / "rdata").glob("*.fjs")),
        "Hurink_sdata": sorted((dataset_root / "Hurink_Data" / "Text" / "sdata").glob("*.fjs")),
        "Hurink_vdata": sorted((dataset_root / "Hurink_Data" / "Text" / "vdata").glob("*.fjs")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=r"c:\Users\shiver\Desktop\Sched_LLM\backend\FJSP_epl\FJSP算例\Monaldo\Fjsp\Job_Data",
    )
    parser.add_argument("--framework-rule", type=str, default="PORTFOLIO")
    parser.add_argument("--traditional-rules", type=str, default="SPT,FIFO,MWKR")
    parser.add_argument("--portfolio-rules", type=str, default="COOP_RH,COOP_RH_ADAPT,TS_GA,CP-SAT")
    parser.add_argument("--hurink-batch-size", type=int, default=20)
    parser.add_argument("--max-cases-per-group", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=r"c:\Users\shiver\Desktop\Sched_LLM\backend\data\fjsp_epl_benchmark")
    parser.add_argument("--call-run-alias", action="store_true")
    parser.add_argument("--skip-brandimarte", action="store_true")
    parser.add_argument("--skip-dauzere", action="store_true")
    parser.add_argument("--skip-hurink", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root 不存在: {dataset_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    framework_rule = str(args.framework_rule).strip().upper()
    traditional_rules = [x.strip().upper() for x in args.traditional_rules.split(",") if x.strip()]
    portfolio_rules = [x.strip().upper() for x in args.portfolio_rules.split(",") if x.strip()]

    groups = collect_case_groups(dataset_root)
    max_cases = int(args.max_cases_per_group)
    if max_cases > 0:
        groups = {k: v[:max_cases] for k, v in groups.items()}

    all_raw: Dict[str, List[Dict[str, Any]]] = {}
    all_main_rows: List[Dict[str, Any]] = []
    all_ext_rows: List[Dict[str, Any]] = []
    group_summary: Dict[str, Any] = {}

    # Brandimarte + Dauzere 全量
    selected_basic_groups: List[str] = []
    if not args.skip_brandimarte:
        selected_basic_groups.append("Brandimarte")
    if not args.skip_dauzere:
        selected_basic_groups.append("Dauzere")
    for name in selected_basic_groups:
        rows = run_cases(
            groups.get(name, []),
            framework_rule=framework_rule,
            traditional_rules=traditional_rules,
            portfolio_rules=portfolio_rules,
            call_run_alias=bool(args.call_run_alias),
        )
        all_raw[name] = rows
        all_main_rows.extend(_expand_main_rows(name, rows))
        all_ext_rows.extend(_expand_extended_rows(name, rows))
        group_summary[name] = _summarize_group(rows)

    # Hurink 分批
    hurink_group_names = [] if args.skip_hurink else ["Hurink_edata", "Hurink_rdata", "Hurink_sdata", "Hurink_vdata"]
    hurink_rows_all: List[Dict[str, Any]] = []
    hurink_batches_meta: Dict[str, Any] = {}
    for name in hurink_group_names:
        files = groups.get(name, [])
        batches = _split_batches(files, int(args.hurink_batch_size))
        this_rows: List[Dict[str, Any]] = []
        batch_meta: List[Dict[str, Any]] = []
        for i, batch in enumerate(batches, start=1):
            batch_rows = run_cases(
                batch,
                framework_rule=framework_rule,
                traditional_rules=traditional_rules,
                portfolio_rules=portfolio_rules,
                call_run_alias=bool(args.call_run_alias),
            )
            this_rows.extend(batch_rows)
            batch_meta.append({"batch_idx": i, "cases": len(batch), "valid_cases": len([x for x in batch_rows if x.get("framework_metrics") and x.get("traditional_best_metrics")])})
        hurink_batches_meta[name] = batch_meta
        all_raw[name] = this_rows
        hurink_rows_all.extend(this_rows)
        all_main_rows.extend(_expand_main_rows(name, this_rows))
        all_ext_rows.extend(_expand_extended_rows(name, this_rows))
        group_summary[name] = _summarize_group(this_rows)

    group_summary["Hurink_overall"] = _summarize_group(hurink_rows_all)
    group_summary["All"] = _summarize_group([x for rows in all_raw.values() for x in rows])

    main_json = out_dir / "main_makespan_results.json"
    ext_json = out_dir / "extended_waiting_results.json"
    raw_json = out_dir / "raw_case_results.json"
    summary_json = out_dir / "summary.json"
    main_csv = out_dir / "main_makespan_results.csv"
    ext_csv = out_dir / "extended_waiting_results.csv"

    raw_json.write_text(json.dumps(all_raw, ensure_ascii=False, indent=2), encoding="utf-8")
    main_json.write_text(json.dumps(all_main_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    ext_json.write_text(json.dumps(all_ext_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json.write_text(
        json.dumps(
            {
                "framework_rule": framework_rule,
                "traditional_rules": traditional_rules,
                "portfolio_rules": portfolio_rules,
                "group_summary": group_summary,
                "hurink_batches": hurink_batches_meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(
        main_csv,
        all_main_rows,
        [
            "group",
            "case",
            "framework_rule",
            "framework_makespan",
            "traditional_rule",
            "traditional_makespan",
            "makespan_improve_pct",
            "framework_is_complete",
            "traditional_is_complete",
            "http_status_full",
        ],
    )
    _write_csv(
        ext_csv,
        all_ext_rows,
        [
            "group",
            "case",
            "framework_transport_time",
            "traditional_transport_time",
            "framework_transport_wait",
            "traditional_transport_wait",
            "framework_sync_loss",
            "traditional_sync_loss",
            "busiest_vehicle_fw",
            "busiest_vehicle_tr",
        ],
    )

    print(json.dumps(group_summary, ensure_ascii=False, indent=2))
    print(f"main_results_json: {main_json}")
    print(f"extended_results_json: {ext_json}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
