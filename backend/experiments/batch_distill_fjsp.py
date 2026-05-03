import json
from pathlib import Path
from typing import Any, Dict, List
import os
import sys
from fastapi.testclient import TestClient

# Ensure we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app

def parse_fjs(file_path: Path) -> Dict[str, Any]:
    raw_lines = [line.strip() for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not raw_lines:
        return {}
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
                cms.append({"machine_id": f"M{m}", "process_time": float(t)})
            operations.append({"op_id": f"J{j + 1}O{o + 1}", "candidate_machines": cms})
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
        {"vehicle_id": "V1", "current_location": "L/U", "speed": 1.0, "load_unload_time": 1.0, "status": "idle"},
        {"vehicle_id": "V2", "current_location": "L/U", "speed": 1.2, "load_unload_time": 1.0, "status": "idle"},
    ]
    return {
        "factory_info": {"factory_id": file_path.stem, "factory_name": "FJSP Bench", "planning_horizon": 50000.0, "current_time": 0.0},
        "shop_floor": {"machines": machines, "vehicles": vehicles, "transport_network": {"nodes": nodes, "edges": edges}},
        "jobs": jobs,
        "simulation_config": {"random_seed": 42, "ppo_max_steps": 500},
        "dispatching_config": {"ppo_policy_id": "latest", "transport_rule": "LOAD_BALANCING"},
    }

def batch_distill():
    client = TestClient(app)
    root = Path(__file__).resolve().parents[1]
    # Target Mk01 to Mk05 for initial distillation to avoid too many LLM calls
    case_dir = root / "FJSP_epl" / "FJSP算例" / "Monaldo" / "Fjsp" / "Job_Data" / "Brandimarte_Data" / "Text"
    case_files = sorted(case_dir.glob("Mk0[1-5].fjs"))
    
    if not case_files:
        print(f"No case files found in {case_dir}")
        return

    print(f"Starting batch distillation for {len(case_files)} cases...")
    
    reports = []
    for i, file_path in enumerate(case_files):
        print(f"[{i+1}/{len(case_files)}] Distilling {file_path.name}...")
        payload = parse_fjs(file_path)
        
        # Call /reflect endpoint which now includes experience saving
        response = client.post("/reflect", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Success! Experience ID: {result.get('experience_id')}")
            reports.append({
                "case": file_path.name,
                "status": "success",
                "experience_id": result.get("experience_id"),
                "reflection_preview": result.get("distilled_experience", "")[:100] + "..."
            })
        else:
            print(f"  Failed! Status: {response.status_code}, Error: {response.text}")
            reports.append({
                "case": file_path.name,
                "status": "failed",
                "error": response.text
            })
            
    out_file = root / "data" / "batch_distillation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(f"\nBatch distillation complete. Report saved to {out_file}")

if __name__ == "__main__":
    batch_distill()
