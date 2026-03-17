from fastapi import APIRouter, HTTPException
from app.models.schema import ScheduleRequest
from app.models.state import build_initial_state
from app.core.simulator import Simulator
from app.core.scheduler import PDR
from app.core.evaluator import Evaluator
from app.llm.ollama_client import OllamaClient
from app.llm.prompt_builder import build_reflection_prompt, build_dispatch_prompt
from app.llm.response_parser import parse_llm_response
from app.llm.memory import memory
from typing import List, Dict, Any

router = APIRouter()
ollama_client = OllamaClient(model="qwen2.5:7b")

@router.post("/run-trajectory")
def run_trajectory(req: ScheduleRequest, rule: str = "SPT"):
    """
    运行基于启发式规则 (PDR) 的完整调度轨迹。
    """
    try:
        initial_state = build_initial_state(req)
        
        def pdr_policy(state):
            return PDR.get_dispatch_action(state, rule=rule)
            
        final_state = Simulator.run_simulation(initial_state, pdr_policy)
        metrics = Evaluator.evaluate(final_state)
        
        if "graph" in final_state:
            del final_state["graph"]
            
        return {
            "rule": rule,
            "metrics": metrics,
            "history_summary": [
                f"{h['job_id']}-{h['op_id']} at {h['machine_id']} ({h['start_time']} to {h['finish_time']})"
                for h in final_state["history"]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run-llm-simulation")
def run_llm_simulation(req: ScheduleRequest):
    """
    运行基于 LLM 决策的完整仿真循环。
    """
    try:
        initial_state = build_initial_state(req)
        
        def llm_policy(state):
            prompt = build_dispatch_prompt(state)
            response = ollama_client.generate(prompt)
            try:
                return parse_llm_response(response)
            except:
                return None # 如果解析失败，模拟器会尝试推进时间
                
        final_state = Simulator.run_simulation(initial_state, llm_policy)
        metrics = Evaluator.evaluate(final_state)
        
        if "graph" in final_state:
            del final_state["graph"]
            
        return {
            "mode": "LLM-Powered",
            "metrics": metrics,
            "history": final_state["history"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 仿真失败: {str(e)}")

@router.post("/reflect")
def reflect_on_trajectories(req: ScheduleRequest):
    """
    ReflecSched 的全局反思接口：对比不同规则的轨迹，生成经验并存入内存。
    """
    try:
        # 1. 运行多个规则的仿真
        trajectories = []
        for rule in ["SPT", "FIFO", "MWKR"]:
            traj = run_trajectory(req, rule)
            # 简化摘要供 Prompt 使用
            traj["history_summary"] = traj["history_summary"][:5] 
            trajectories.append(traj)
            
        # 2. 构建反思提示词
        prompt = build_reflection_prompt(trajectories)
        
        # 3. 调用 LLM 进行反思分析
        reflection_result = ollama_client.generate(prompt)
        
        # 4. 将提炼的经验存入内存单例中
        # 简单处理：将整个返回文本作为一个条目，或者按行分割
        memory.add_experience(reflection_result)
        
        return {
            "trajectories_metrics": [t["metrics"] for t in trajectories],
            "reflection": reflection_result,
            "memory_count": len(memory.get_all())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"反思失败: {str(e)}")
