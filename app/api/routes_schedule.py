from fastapi import APIRouter, HTTPException
from app.models.schema import ScheduleRequest
from app.models.state import build_initial_state
from app.llm.ollama_client import OllamaClient
from app.llm.prompt_builder import build_dispatch_prompt
from app.llm.response_parser import parse_llm_response

router = APIRouter()

ollama_client = OllamaClient(model="qwen2.5:7b")


@router.post("/run")
def run_schedule(req: ScheduleRequest):
    try:
        state = build_initial_state(req)
        prompt = build_dispatch_prompt(state)
        llm_text = ollama_client.generate(prompt)
        decision = parse_llm_response(llm_text)

        return {
            "message": "调度建议生成成功",
            "decision": decision,
            "raw_response": llm_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))