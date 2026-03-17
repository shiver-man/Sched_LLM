from pydantic import BaseModel
from typing import List, Optional

class CandidateMachine(BaseModel):
    machine_id: str
    process_time: float

class Operation(BaseModel):
    op_id: str
    candidate_machines: List[CandidateMachine]

class Job(BaseModel):
    job_id: str
    release_time: float
    due_time: float
    operations: List[Operation]

class Machine(BaseModel):
    machine_id: str
    machine_type: str
    location: str

class Vehicle(BaseModel):
    vehicle_id: str
    current_location: str
    speed: float
    capacity: int

class LayoutNode(BaseModel):
    node_id: str

class LayoutEdge(BaseModel):
    from_node: str
    to_node: str
    distance: float

class Layout(BaseModel):
    nodes: List[LayoutNode]
    edges: List[LayoutEdge]

class Objective(BaseModel):
    type: str
    weight: float

class ScheduleRequest(BaseModel):
    jobs: List[Job]
    machines: List[Machine]
    vehicles: List[Vehicle]
    layout: Layout
    current_time: float
    objective: Objective
