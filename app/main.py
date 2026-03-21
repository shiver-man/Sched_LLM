from fastapi import FastAPI
from app.config import settings
from app.api.routes_simulation import router as simulation_router

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="用户参数驱动的生产-运输一体化柔性作业车间调度系统",
)

app.include_router(simulation_router, prefix="/simulation", tags=["simulation"])


@app.get("/")
def root():
    return {
        "message": "Sched_LLM is running",
        "model": settings.ollama_model,
        "ollama_base_url": settings.ollama_base_url,
    }
