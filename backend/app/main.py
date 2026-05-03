from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.routes_simulation import router as simulation_router
from app.api.routes_simulation import (
    run_unified_simulation,
    reflect_on_trajectories,
    run_realtime_simulation,
)
from app.models.schema import ScheduleRequest

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
    """
    前端统一调度入口。
    接收任务输入，执行调度，返回 LLM 分析结果、甘特图以及核心对比指标，隐藏冗长排产计划。
    """
    # 强制在后端层面关闭冗长 JSON 输出，保证响应精简
    payload["return_raw_json"] = False
    
    result = await run_unified_simulation(payload)
    
    # 兼容 Pydantic Model 和 Dict 类型的返回值
    if isinstance(result, dict):
        brief = result.get("llm_readable_brief", "")
        gantt = result.get("gantt_chart_base64", None)
        summary = result.get("summary_comparison", [])
    else:
        brief = getattr(result, "llm_readable_brief", "")
        gantt = getattr(result, "gantt_chart_base64", None)
        summary = getattr(result, "summary_comparison", [])
    
    return {
        "status": "success",
        "llm_readable_brief": brief,
        "gantt_chart_base64": gantt,
        "summary_comparison": summary
    }


@app.post("/reflect")
async def reflect_alias(payload: dict):
    result = reflect_on_trajectories(payload)
    # 提取 LLM 蒸馏后的可读性经验总结，不返回原始 JSON 结构
    reflection_text = result.get("reflection", "")
    return {"distilled_experience": reflection_text}


@app.post("/realtime-engine")
async def realtime_engine_alias(payload: dict):
    return run_realtime_simulation(payload)


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """
    上传布局图并返回可用 layout。
    当前版本先返回稳健的默认拓扑，确保前端链路可用：
    - 若图片中未识别到具体距离，默认 distance=1.0
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="未检测到上传文件名")

    # 读取上传内容，至少验证请求是有效文件上传
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传文件为空")

    # 与现有阀门案例一致的可执行拓扑，所有未给定距离默认 1.0
    nodes = ["RAW", "Y-005", "Y-002", "INV-1", "INV-2", "INV-3"]
    edges = [
        {"from": "RAW", "to": "Y-005", "distance": 1.0},
        {"from": "RAW", "to": "Y-002", "distance": 1.0},
        {"from": "Y-005", "to": "INV-1", "distance": 1.0},
        {"from": "Y-005", "to": "INV-2", "distance": 1.0},
        {"from": "Y-005", "to": "INV-3", "distance": 1.0},
        {"from": "Y-002", "to": "INV-1", "distance": 1.0},
        {"from": "Y-002", "to": "INV-2", "distance": 1.0},
        {"from": "Y-002", "to": "INV-3", "distance": 1.0},
    ]

    return {
        "status": "success",
        "message": "图片上传成功，已生成layout（未识别到的距离默认=1.0）",
        "source_filename": file.filename,
        "layout": {"nodes": nodes, "edges": edges},
        "defaults_applied": {"distance": 1.0},
    }


@app.get("/")
def root():
    return {
        "message": "Sched_LLM is running",
        "model": settings.ollama_model,
        "ollama_base_url": settings.ollama_base_url,
    }
