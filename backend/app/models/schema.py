from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, AliasChoices, field_validator
from app.models.job import Job
from app.models.machine import Machine
from app.models.transport import Vehicle


class LayoutEdge(BaseModel):
    from_node: str = Field(
        ...,
        validation_alias=AliasChoices("from", "from_node"),
        serialization_alias="from",
    )
    to_node: str = Field(
        ...,
        validation_alias=AliasChoices("to", "to_node"),
        serialization_alias="to",
    )
    distance: float = Field(..., gt=0)

    model_config = {
        "populate_by_name": True
    }


class Layout(BaseModel):
    nodes: List[str]
    edges: List[LayoutEdge]
    directed: bool = False

    @field_validator("nodes", mode="before")
    @classmethod
    def normalize_nodes(cls, value):
        if not isinstance(value, list):
            return value
        normalized = []
        for node in value:
            if isinstance(node, dict):
                normalized.append(node.get("node_id"))
            else:
                normalized.append(node)
        return normalized


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


class ScheduleStep(BaseModel):
    step: int
    job_id: str
    op_id: str
    machine_id: str
    vehicle_id: Optional[str] = None
    transport_time: float = 0.0
    start_time: float
    finish_time: float
    lookahead_score: Optional[float] = None


class ScheduleMetrics(BaseModel):
    makespan: float
    utilization: float
    total_tardiness: float
    total_transport_time: float
    num_events: int
    is_complete: bool = False
    total_ops_expected: int = 0
    vehicle_utilization: float = 0.0
    transport_wait_time: float = 0.0
    busiest_vehicle: Optional[str] = None
    busiest_path: Optional[str] = None
    path_conflicts: int = 0
    machine_idle_reasons: Optional[Dict[str, int]] = None


class ScheduleScheme(BaseModel):
    category: str
    rule: str
    metrics: ScheduleMetrics
    plan: List[ScheduleStep]


class MultiStrategyResponse(BaseModel):
    status: str = "success"
    detailed_schemes: List[ScheduleScheme]
    summary_comparison: List[Dict[str, Any]]
    llm_readable_brief: str = ""


class SimulationRuleRequest(ScheduleRequest):
    rule: str = "SPT"
    max_steps: int = 1000


class SchedulePlanRequest(ScheduleRequest):
    rules: List[str] = ["SPT", "FIFO", "MWKR"]
    objective: str = "makespan"
    max_steps: int = 1000


class FailureRecoveryRequest(SchedulePlanRequest):
    fault_time: float = Field(0.0, ge=0)
    candidate_failed_machines: List[str] = []
    max_failed_machines: int = Field(1, ge=1)
    include_no_failure: bool = True
    max_scenarios: int = Field(256, ge=1)


class PPOTrainRequest(ScheduleRequest):
    episodes: int = Field(200, ge=1)
    max_steps: int = Field(1000, ge=1)
    gamma: float = Field(0.99, gt=0, le=1)
    clip_ratio: float = Field(0.2, gt=0, lt=1)
    learning_rate: float = Field(0.01, gt=0)
    update_epochs: int = Field(4, ge=1)
    process_time_noise_low: float = Field(0.8, gt=0)
    process_time_noise_high: float = Field(1.2, gt=0)
    seed: int = 42


class PPOPlanRequest(ScheduleRequest):
    policy_id: str = "latest"
    max_steps: int = Field(1000, ge=1)
    process_time_noise_low: float = Field(0.8, gt=0)
    process_time_noise_high: float = Field(1.2, gt=0)
    seed: int = 123


class ProcessingTimeUncertainty(BaseModel):
    fluctuation_low: float = Field(0.9, ge=0)
    fluctuation_high: float = Field(1.1, ge=0)
    distribution_type: str = Field("uniform", description="uniform / normal")


class TransportTimeUncertainty(BaseModel):
    fluctuation_low: float = Field(0.95, ge=0)
    fluctuation_high: float = Field(1.15, ge=0)
    distribution_type: str = Field("uniform", description="uniform / normal")


class MachineBreakdownModel(BaseModel):
    enabled: bool = True
    breakdown_probability: float = Field(0.1, ge=0, le=1)
    mean_time_to_failure: float = Field(150.0, gt=0)
    mean_repair_time: float = Field(20.0, gt=0)


class VehicleDelayModel(BaseModel):
    enabled: bool = True
    delay_probability: float = Field(0.05, ge=0, le=1)
    delay_range: List[float] = Field([2.0, 10.0], description="[min_delay, max_delay]")


class UncertaintyConfig(BaseModel):
    processing: ProcessingTimeUncertainty = Field(default_factory=ProcessingTimeUncertainty)
    transport: TransportTimeUncertainty = Field(default_factory=TransportTimeUncertainty)
    breakdown: MachineBreakdownModel = Field(default_factory=MachineBreakdownModel)
    vehicle_delay: VehicleDelayModel = Field(default_factory=VehicleDelayModel)


class DynamicUncertaintyRequest(ScheduleRequest):
    uncertainty_config: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    num_scenarios: int = Field(5, ge=1, description="需要模拟的代表性场景数量")
    max_steps: int = Field(1000, ge=1)
    policy_type: str = Field("PPO", description="PPO / SPT / FIFO / MWKR")
    policy_id: Optional[str] = None
    seed: int = 42
