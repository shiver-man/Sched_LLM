from typing import Optional
from pydantic import BaseModel, Field


class Machine(BaseModel):
    machine_id: str = Field(..., description="机器ID")
    machine_type: Optional[str] = Field(None, description="机器类型")
    location: str = Field(..., description="机器所在位置")
    status: str = Field("idle", description="idle / busy / down")
    available_time: float = Field(0.0, ge=0)
    current_job: Optional[str] = None