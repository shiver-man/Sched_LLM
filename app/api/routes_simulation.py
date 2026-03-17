from fastapi import APIRouter, HTTPException
from app.models.schema import ScheduleRequest
from app.models.state import build_initial_state
from app.core.simulator import Simulator
from app.core.scheduler import PDR
from app.core.evaluator import Evaluator
from app.llm.ollama_client import OllamaClient
from app.llm.prompt_builder import build_reflection_prompt
from typing import List, Dict, Any

router = APIRouter()
ollama_client = OllamaClient(model="qwen2.5:7b")

@router.post("/run-trajectory")
def run_trajectory(req: ScheduleRequest, rule: str = "SPT"):
    """
    运行完整的调度轨迹并返回评估结果。
    """
    try:
        # 1. 初始化状态
        initial_state = build_initial_state(req)
        
        # 2. 定义策略函数 (基于 PDR 规则)
        def pdr_policy(state):
            return PDR.get_dispatch_action(state, rule=rule)
            
        # 3. 运行仿真
        final_state = Simulator.run_simulation(initial_state, pdr_policy)
        
        # 4. 评估结果
        metrics = Evaluator.evaluate(final_state)
        
        # 移除不可序列化的 graph
        if "graph" in final_state:
            del final_state["graph"]
            
        return {
            "rule": rule,
            "metrics": metrics,
            "history_summary": [
                f"{h['job_id']}-{h['op_id']} at {h['machine_id']} ({h['start_time']} to {h['finish_time']})"
                for h in final_state["history"][:10] # 只返回前10条摘要
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reflect")
def reflect_on_trajectories(req: ScheduleRequest):
    """
    ReflecSched 的全局反思接口：对比不同规则的轨迹并生成高层策略。
    """
    try:
        # 1. 运行多个规则的仿真
        trajectories = []
        for rule in ["SPT", "FIFO", "MWKR"]:
            traj = run_trajectory(req, rule)
            trajectories.append(traj)
            
        # 2. 构建反思提示词
        prompt = build_reflection_prompt(trajectories)
        
        # 3. 调用 LLM 进行反思分析
        reflection_result = ollama_client.generate(prompt)
        
        return {
            "trajectories": trajectories,
            "reflection": reflection_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"反思失败: {str(e)}")
