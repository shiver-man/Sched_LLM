from fastapi import APIRouter, HTTPException
import traceback

from app.models.schema import ScheduleRequest, FailureRecoveryRequest
from app.api.routes_simulation import generate_failure_recovery_plan

router = APIRouter()


def run_schedule(req: ScheduleRequest):
    try:
        metadata = req.metadata or {}
        failure_req = FailureRecoveryRequest(
            **req.model_dump(),
            rules=metadata.get("rules", ["SPT", "FIFO", "MWKR"]),
            objective=metadata.get("objective", "makespan"),
            max_steps=metadata.get("max_steps", 1000),
            fault_time=metadata.get("fault_time", req.current_time),
            candidate_failed_machines=metadata.get("candidate_failed_machines", []),
            max_failed_machines=metadata.get("max_failed_machines", 1),
            include_no_failure=metadata.get("include_no_failure", True),
            max_scenarios=metadata.get("max_scenarios", 256),
        )
        return generate_failure_recovery_plan(failure_req)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"调度运行失败: {str(e)}")
