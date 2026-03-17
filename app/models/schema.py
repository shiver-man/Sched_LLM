from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class CandidateMachine(BaseModel):
    machine_id: str
    process_time: int


class OperationInput(BaseModel):
    op_id: str
    candidate_machines: List[CandidateMachine]


class JobInput(BaseModel):
    job_id: str
    operations: List[OperationInput]
    release_time: int = 0
    due_time: Optional[int] = None


class MachineInput(BaseModel):
    machine_id: str
    machine_type: Optional[str] = None
    location: str
    status: str = "idle"


class VehicleInput(BaseModel):
    vehicle_id: str
    current_location: str
    speed: float = 1.0
    capacity: int = 1
    status: str = "idle"


class EdgeInput(BaseModel):
    from_node: str
    to_node: str
    distance: float


class LayoutInput(BaseModel):
    nodes: List[str]
    edges: List[EdgeInput]


class ObjectiveInput(BaseModel):
    minimize_makespan: bool = True
    minimize_transport_time: bool = True
    minimize_tardiness: bool = False


class ScheduleRequest(BaseModel):
    jobs: List[JobInput]
    machines: List[MachineInput]
    vehicles: List[VehicleInput]
    layout: LayoutInput
    objective: ObjectiveInput = Field(default_factory=ObjectiveInput)
    current_time: int = 0