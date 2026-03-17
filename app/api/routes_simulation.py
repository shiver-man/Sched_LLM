from fastapi import APIRouter, HTTPException
from app.models.schema import SimulationRuleRequest, ScheduleRequest
from app.models.state import build_initial_state
from app.core.simulator import Simulator
from app.core.scheduler import PDR
from app.core.evaluator import Evaluator
from app.llm.ollama_client import OllamaClient
from app.llm.prompt_builder import build_reflection_prompt
from app.config import settings

router = APIRouter()
ollama_client = OllamaClient(model=settings.ollama_model)

@router.post("/run-trajectory")
def run_trajectory(req: SimulationRuleRequest):
    try:
        initial_state = build_initial_state(req)

        def pdr_policy(state):
            return PDR.get_dispatch_action(state, rule=req.rule)

        final_state = Simulator.run_simulation(initial_state, pdr_policy, max_steps=req.max_steps)
        metrics = Evaluator.evaluate(final_state)

        history_summary = [
            {
                "job_id": h["job_id"],
                "op_id": h["op_id"],
                "machine_id": h["machine_id"],
                "vehicle_id": h.get("vehicle_id"),
                "transport_time": h.get("transport_time", 0),
                "start_time": h["start_time"],
                "finish_time": h["finish_time"],
            }
            for h in final_state["history"]
        ]

        return {
            "rule": req.rule,
            "metrics": metrics,
            "history_summary": history_summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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