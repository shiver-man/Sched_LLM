from fastapi import FastAPI
from app.api.routes_schedule import router as schedule_router

app = FastAPI(title="Sched_LLM Dynamic Scheduling System")

app.include_router(schedule_router, prefix="/schedule", tags=["schedule"])


@app.get("/")
def root():
    return {"message": "Sched_LLM is running"}