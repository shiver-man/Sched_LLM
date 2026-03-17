from fastapi import APIRouter, HTTPException
from app.models.schema import SimulationRuleRequest, ScheduleRequest, SchedulePlanRequest
from app.models.state import build_initial_state
from app.core.simulator import Simulator
from app.core.scheduler import PDR
from app.core.evaluator import Evaluator
from app.llm.ollama_client import OllamaClient
from app.llm.prompt_builder import build_reflection_prompt
from app.config import settings

router = APIRouter()
ollama_client = OllamaClient(model=settings.ollama_model)


def _run_rule_plan(req: ScheduleRequest, rule: str, max_steps: int):
    initial_state = build_initial_state(req)

    def pdr_policy(state):
        return PDR.get_dispatch_action(state, rule=rule)

    final_state = Simulator.run_simulation(initial_state, pdr_policy, max_steps=max_steps)
    metrics = Evaluator.evaluate(final_state)
    plan = [
        {
            "step": idx + 1,
            "job_id": h["job_id"],
            "op_id": h["op_id"],
            "machine_id": h["machine_id"],
            "vehicle_id": h.get("vehicle_id"),
            "transport_time": h.get("transport_time", 0),
            "start_time": h["start_time"],
            "finish_time": h["finish_time"],
        }
        for idx, h in enumerate(final_state["history"])
    ]

    return {"rule": rule, "metrics": metrics, "plan": plan}


def _pick_best(results, objective: str):
    objective = objective.lower()
    if objective == "utilization":
        return max(results, key=lambda x: x["metrics"]["utilization"])
    if objective == "total_tardiness":
        return min(results, key=lambda x: x["metrics"]["total_tardiness"])
    if objective == "total_transport_time":
        return min(results, key=lambda x: x["metrics"]["total_transport_time"])
    if objective == "num_events":
        return min(results, key=lambda x: x["metrics"]["num_events"])
    return min(results, key=lambda x: x["metrics"]["makespan"])


@router.post("/run-trajectory")
def run_trajectory(req: SimulationRuleRequest):
    try:
        result = _run_rule_plan(req, req.rule, req.max_steps)

        return {
            "rule": result["rule"],
            "metrics": result["metrics"],
            "history_summary": result["plan"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plan")
def generate_schedule_plan(req: SchedulePlanRequest):
    try:
        requested_rules = [rule.upper() for rule in req.rules]
        supported_rules = ["SPT", "FIFO", "MWKR"]
        rules = [rule for rule in requested_rules if rule in supported_rules]
        if not rules:
            rules = supported_rules

        results = [_run_rule_plan(req, rule, req.max_steps) for rule in rules]
        best = _pick_best(results, req.objective)

        return {
            "status": "success",
            "objective": req.objective,
            "evaluated_rules": rules,
            "best_rule": best["rule"],
            "best_metrics": best["metrics"],
            "best_schedule_plan": best["plan"],
            "all_rule_results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调度方案生成失败: {str(e)}")


@router.post("/reflect")
def reflect_on_trajectories(req: ScheduleRequest):
    try:
        trajectories = []
        for rule in ["SPT", "FIFO", "MWKR"]:
            tmp_req = SimulationRuleRequest(**req.model_dump(), rule=rule, max_steps=1000)
            result = run_trajectory(tmp_req)
            trajectories.append(result)

        prompt = build_reflection_prompt(trajectories)
        reflection_result = ollama_client.generate(prompt)
        return {
            "trajectories": trajectories,
            "reflection": reflection_result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"反思失败: {str(e)}")
