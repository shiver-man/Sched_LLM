from fastapi import APIRouter, HTTPException
import traceback

from app.models.schema import ScheduleRequest
from app.models.state import build_initial_state
from app.llm.ollama_client import OllamaClient
from app.llm.prompt_builder import build_dispatch_prompt
from app.llm.response_parser import parse_llm_response
from app.core.dispatcher import Dispatcher
from app.config import settings

router = APIRouter()
ollama_client = OllamaClient(model=settings.ollama_model)


@router.post("/run")
def run_schedule(req: ScheduleRequest):
    try:
        state = build_initial_state(req)
        prompt = build_dispatch_prompt(state, req.strategic_experience)
        llm_text = ollama_client.generate(prompt)
        decision = parse_llm_response(llm_text)
        updated_state = Dispatcher.apply_decision(state, decision)

        response_state = dict(updated_state)
        if "graph" in response_state:
            del response_state["graph"]

        return {
            "status": "success",
            "decision": decision,
            "new_state": response_state,
            "raw_llm_output": llm_text,
            "prompt": prompt,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"调度运行失败: {str(e)}")