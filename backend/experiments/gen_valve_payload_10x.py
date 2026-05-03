import json
from pathlib import Path

base = Path(r"c:\Users\shiver\Desktop\Sched_LLM\backend\data")
proc = json.loads((base / "valve_processes.json").read_text(encoding="utf-8"))
routes = proc["routes"]

jobs = []
for route in routes:
    qty = int(route.get("quantity", 1))
    for i in range(1, qty + 1):
        job_id = f"{route['route_id']}_{i:02d}"
        ops = []
        for k, op in enumerate(route["operations"], start=1):
            ops.append({
                "operation_id": f"{job_id}_OP{k}",
                "candidate_machines": [
                    {
                        "machine_id": op["machine_id"],
                        "processing_time": float(op["processing_time_min"]),
                    }
                ],
            })
        jobs.append(
            {
                "job_id": job_id,
                "release_time": 0.0,
                "due_time": 9999.0,
                "initial_location": "RAW",
                "metadata": {
                    "product_type": route["product_type"],
                    "route_id": route["route_id"],
                    "instance_index": i,
                    "part_name": "+".join([o["part_name"] for o in route["operations"]]),
                    "process_content": " -> ".join([o["process_content"] for o in route["operations"]]),
                },
                "operations": ops,
            }
        )

payload = {
    "factory_info": {
        "factory_id": "valve_factory_case_1x",
        "factory_name": "Valve Factory 1x",
        "planning_horizon": 100000.0,
        "current_time": 0.0,
    },
    "shop_floor": {
        "machines": [
            {"machine_id": "Y-005", "type": "VERTICAL_CNC", "location": "Y-005", "status": "idle"},
            {"machine_id": "Y-002", "type": "HORIZONTAL_CNC", "location": "Y-002", "status": "idle"},
        ],
        "vehicles": [
            {"vehicle_id": "V1", "start_location": "RAW", "speed": 1.0, "load_unload_time": 1.0, "status": "idle"},
            {"vehicle_id": "V2", "start_location": "RAW", "speed": 1.0, "load_unload_time": 1.0, "status": "idle"},
        ],
        "transport_network": {
            "nodes": ["RAW", "Y-005", "Y-002", "INV-1", "INV-2", "INV-3"],
            "edges": [
                {"from": "RAW", "to": "Y-005", "distance": 1.0},
                {"from": "RAW", "to": "Y-002", "distance": 1.0},
                {"from": "Y-005", "to": "INV-1", "distance": 1.0},
                {"from": "Y-005", "to": "INV-2", "distance": 1.0},
                {"from": "Y-005", "to": "INV-3", "distance": 1.0},
                {"from": "Y-002", "to": "INV-1", "distance": 1.0},
                {"from": "Y-002", "to": "INV-2", "distance": 1.0},
                {"from": "Y-002", "to": "INV-3", "distance": 1.0},
            ],
        },
    },
    "jobs": jobs,
    "simulation_config": {"random_seed": 42, "ppo_max_steps": 5000},
    "dispatching_config": {"ppo_policy_id": "latest", "transport_rule": "LOAD_BALANCING"},
    "llm_config": {"use_ollama": False},
    "return_raw_json": True,
}

out = base / "valve_experiment_payload_1x.json"
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved", out)
print("jobs", len(jobs))
