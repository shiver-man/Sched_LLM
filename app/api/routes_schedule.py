from fastapi import APIRouter, HTTPException
from app.models.schema import ScheduleRequest
from app.models.state import build_initial_state
from app.llm.ollama_client import OllamaClient
from app.llm.prompt_builder import build_dispatch_prompt
from app.llm.response_parser import parse_llm_response
from app.core.dispatcher import Dispatcher

router = APIRouter()

# 初始化 Ollama 客户端
ollama_client = OllamaClient(model="qwen2.5:7b")

@router.post("/run")
def run_schedule(req: ScheduleRequest):
    """
    接收用户输入的调度参数，调用 LLM 生成决策，并执行该决策。
    """
    try:
        # 1. 根据用户输入构造内部状态
        state = build_initial_state(req)
        
        # 2. 将状态转换为 Prompt
        prompt = build_dispatch_prompt(state)
        
        # 3. 调用 Ollama 获取调度建议
        llm_text = ollama_client.generate(prompt)
        
        # 4. 解析 LLM 返回的 JSON 决策
        decision = parse_llm_response(llm_text)
        
        # 5. 执行决策并更新状态
        updated_state = Dispatcher.apply_decision(state, decision)
        
        # 移除不可序列化的 graph 对象以便返回 JSON
        if "graph" in updated_state:
            del updated_state["graph"]
            
        return {
            "status": "success",
            "decision": decision,
            "new_state": updated_state,
            "raw_llm_output": llm_text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调度运行失败: {str(e)}")
