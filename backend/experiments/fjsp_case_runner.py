import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
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
    }


def run_cases(case_files: List[Path]) -> List[Dict[str, Any]]:
    client = TestClient(app)
    results = []
    for file_path in case_files:
        payload = parse_fjs(file_path)
        resp = client.post("/run", json=payload)
        entry: Dict[str, Any] = {"case": str(file_path), "http_status": resp.status_code}
        if resp.status_code != 200:
            entry["error"] = resp.text
            results.append(entry)
            continue
        full = client.post("/simulation/run", json=payload)
        entry["http_status_full"] = full.status_code
        if full.status_code != 200:
            entry["error"] = full.text
            results.append(entry)
            continue
        body = full.json()
        summary = body.get("summary_comparison", [])
        entry["best_rule"] = summary[0]["rule"] if summary else None
        entry["best_makespan"] = summary[0]["makespan"] if summary else None
        entry["num_schemes"] = len(body.get("detailed_schemes", []))
        ga = next((x for x in summary if x.get("rule") == "GA"), None)
        entry["ga_makespan"] = ga.get("makespan") if ga else None
        entry["ga_complete"] = ga.get("is_complete") if ga else None
        entry["llm_brief_len"] = len(body.get("llm_readable_brief", ""))
        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str, default="FJSP算例/FJSP算例/Monaldo/Fjsp/Job_Data/Brandimarte_Data/Text/Mk01.fjs")
    parser.add_argument("--max-cases", type=int, default=1)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset_root = root.parent
    cases = sorted(dataset_root.glob(args.glob))[: max(1, args.max_cases)]
    results = run_cases(cases)
    out_file = root / "data" / "fjsp_backend_test_report.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"report_saved_to: {out_file}")


if __name__ == "__main__":
    main()
