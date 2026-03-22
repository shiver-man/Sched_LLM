import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.schema import SchedulePlanRequest
from app.models.state import build_initial_state
from app.core.simulator import Simulator
from app.core.scheduler import PDR
from app.core.evaluator import Evaluator


def load_data() -> SchedulePlanRequest:
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    with open(os.path.join(data_dir, "jobs.json"), "r") as f:
        jobs = json.load(f)
    with open(os.path.join(data_dir, "machines.json"), "r") as f:
        machines = json.load(f)
    with open(os.path.join(data_dir, "transport.json"), "r") as f:
        vehicles = json.load(f)
    with open(os.path.join(data_dir, "layout.json"), "r") as f:
        layout = json.load(f)

    req = SchedulePlanRequest(
        jobs=jobs,
        machines=machines,
        vehicles=vehicles,
        layout=layout,
        current_time=0.0,
        objective="makespan",
        rules=["SPT", "FIFO", "MWKR"],
        max_steps=1000,
    )
    return req


def run_rule(req: SchedulePlanRequest, rule: str):
    def pdr_policy(state):
        return PDR.get_dispatch_action(state, rule=rule)

    final_state = Simulator.run_simulation(build_initial_state(req), pdr_policy, max_steps=req.max_steps)
    metrics = Evaluator.evaluate(final_state)
    plan = [
        {
            "step": idx + 1,
            "job_id": h["job_id"],
            "op_id": h["op_id"],
            "machine_id": h["machine_id"],
            "vehicle_id": h.get("vehicle_id"),
            "start_time": h["start_time"],
            "finish_time": h["finish_time"],
            "transport_time": h.get("transport_time", 0),
        }
        for idx, h in enumerate(final_state["history"])
    ]
    return {"rule": rule, "metrics": metrics, "plan": plan}


def pick_best(results, objective: str):
    objective = objective.lower()
    if objective == "utilization":
        return max(results, key=lambda x: x["metrics"]["utilization"])
    if objective == "total_tardiness":
        return min(results, key=lambda x: x["metrics"]["total_tardiness"])
    if objective == "total_transport_time":
        return min(results, key=lambda x: x["metrics"]["total_transport_time"])
    return min(results, key=lambda x: x["metrics"]["makespan"])


def run_baseline_experiment():
    req = load_data()
    results = [run_rule(req, rule.upper()) for rule in req.rules]
    best = pick_best(results, req.objective)

    output = {
        "objective": req.objective,
        "best_rule": best["rule"],
        "best_metrics": best["metrics"],
        "best_schedule_plan": best["plan"],
        "all_rule_results": results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    run_baseline_experiment()
