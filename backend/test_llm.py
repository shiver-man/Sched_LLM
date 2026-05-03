import asyncio
import json
from app.api.routes_simulation import run_unified_simulation
from experiments.fjsp_case_runner import parse_fjs
from pathlib import Path

async def test():
    mk01_path = Path("FJSP_epl") / "FJSP算例" / "Monaldo" / "Fjsp" / "Job_Data" / "Brandimarte_Data" / "Text" / "Mk01.fjs"
    payload = parse_fjs(mk01_path)
    payload["llm_config"] = {"use_ollama": True}
    payload["return_raw_json"] = False
    
    print("Starting simulation and LLM analysis (this may take a few seconds)...")
    result = await run_unified_simulation(payload)
    
    if isinstance(result, dict):
        brief = result.get("llm_readable_brief", "")
    else:
        brief = getattr(result, "llm_readable_brief", "")
        
    print("\n" + "="*50)
    print("LLM RESPONSE (Testing New Prompt):")
    print("="*50)
    print(brief)
    print("="*50)

if __name__ == "__main__":
    asyncio.run(test())
