from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.models.job import Job
from app.models.machine import Machine
from app.models.transport import Vehicle


class LayoutEdge(BaseModel):
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    distance: float = Field(..., gt=0)

    model_config = {
        "populate_by_name": True
    }


class Layout(BaseModel):
    nodes: List[str]
    edges: List[LayoutEdge]
    directed: bool = False


class ScheduleRequest(BaseModel):
    jobs: List[Job]
    machines: List[Machine]
    vehicles: List[Vehicle] = []
    layout: Layout
    current_time: float = 0.0
    strategic_experience: str = ""
    metadata: Optional[Dict[str, Any]] = None


class Decision(BaseModel):
    job_id: str
    op_id: str
    machine_id: str
    vehicle_id: Optional[str] = None
    reason: str = ""


class SimulationRuleRequest(ScheduleRequest):
    rule: str = "SPT"
    max_steps: int = 1000