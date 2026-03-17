from typing import List, Optional
from pydantic import BaseModel, Field


class CandidateMachine(BaseModel):
    machine_id: str = Field(..., description="候选机器ID")
    process_time: float = Field(..., ge=0, description="在该机器上的加工时间")


class Operation(BaseModel):
    op_id: str = Field(..., description="工序ID")
    source_location: Optional[str] = Field(None, description="该工序开始前工件所在位置，可为空")
    candidate_machines: List[CandidateMachine]


class Job(BaseModel):
    job_id: str = Field(..., description="工件ID")
    operations: List[Operation]
    release_time: float = Field(0.0, ge=0)
    due_time: float = Field(10**9, ge=0)
    initial_location: str = Field(..., description="工件初始位置")