from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.routes_simulation import router as simulation_router
from app.api.routes_simulation import run_unified_simulation

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="用户参数驱动的生产-运输一体化柔性作业车间调度系统",
)

app.include_router(simulation_router, prefix="/simulation", tags=["simulation"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/run")
async def run_alias(payload: dict):
    result = await run_unified_simulation(payload)
    if isinstance(result, dict):
        brief = result.get("llm_readable_brief", "")
    else:
        brief = getattr(result, "llm_readable_brief", "")
    return {"llm_readable_brief": brief}


@app.get("/")
def root():
    return {
        "message": "Sched_LLM is running",
        "model": settings.ollama_model,
        "ollama_base_url": settings.ollama_base_url,
    }
