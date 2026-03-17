from typing import Optional
from pydantic import BaseModel, Field


class Vehicle(BaseModel):
    vehicle_id: str = Field(..., description="运输车ID")
    current_location: str = Field(..., description="当前位置")
    speed: float = Field(1.0, gt=0, description="速度")
    capacity: int = Field(1, ge=1, description="容量")
    load_unload_time: float = Field(0.0, ge=0, description="装卸时间")
    status: str = Field("idle", description="idle / busy")
    available_time: float = Field(0.0, ge=0)
    current_task: Optional[str] = None