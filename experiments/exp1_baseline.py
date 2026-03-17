import json
import os
import sys

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.schema import ScheduleRequest, Job, Machine, Vehicle, Layout, Objective
from app.models.state import build_initial_state
from app.core.simulator import Simulator
from app.core.scheduler import PDR
from app.core.evaluator import Evaluator
from app.llm.ollama_client import OllamaClient
from app.llm.prompt_builder import build_dispatch_prompt, build_reflection_prompt
from app.llm.response_parser import parse_llm_response

def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    with open(os.path.join(data_dir, "jobs.json"), "r") as f:
        jobs = json.load(f)
    with open(os.path.join(data_dir, "machines.json"), "r") as f:
        machines = json.load(f)
    with open(os.path.join(data_dir, "transport.json"), "r") as f:
        vehicles = json.load(f)
    with open(os.path.join(data_dir, "layout.json"), "r") as f:
        layout = json.load(f)
        
    req = ScheduleRequest(
        jobs=jobs,
        machines=machines,
        vehicles=vehicles,
        layout=layout,
        current_time=0.0,
        objective=Objective(type="makespan", weight=1.0)
    )
    return req

def run_baseline_experiment():
    print("=== Step 1: Loading Data ===")
    req = load_data()
    initial_state = build_initial_state(req)
    
    ollama = OllamaClient(model="qwen2.5:7b")

    print("\n=== Step 2: Generating Trajectories with PDR Rules ===")
    rules = ["SPT", "FIFO", "MWKR"]
    trajectories = []
    
    for rule in rules:
        print(f"Running simulation with rule: {rule}...")
        # 定义 PDR 策略函数
        def pdr_policy(state):
            return PDR.get_dispatch_action(state, rule=rule)
            
        final_state = Simulator.run_simulation(initial_state, pdr_policy)
        metrics = Evaluator.evaluate(final_state)
        
        trajectories.append({
            "rule": rule,
            "metrics": metrics,
            "history_summary": [f"{h['job_id']}-{h['op_id']}@{h['machine_id']}" for h in final_state["history"][:5]]
        })
        print(f"Result for {rule}: Makespan={metrics['makespan']}, Utilization={metrics['utilization']}")

    print("\n=== Step 3: Hierarchical Reflection (LLM Analysis) ===")
    reflection_prompt = build_reflection_prompt(trajectories)
    print("Calling LLM for reflection...")
    strategic_experience = ollama.generate(reflection_prompt)
    print("\n[Strategic Experience from LLM]:")
    print(strategic_experience)

    print("\n=== Step 4: Final Scheduling with LLM Decision + Reflection ===")
    # 这里的策略函数调用 LLM，并传入 reflection 得到的经验
    def llm_policy(state):
        # 只有在有可用资源时才调用 LLM，否则返回 None 让仿真器推进时间
        available_machines = [m for m in state["machines"] if m["status"] == "idle"]
        dispatchable_jobs = [j for j in state["jobs"] if not j["finished"] and state["time"] >= j["release_time"]]
        
        if not available_machines or not dispatchable_jobs:
            return None
            
        prompt = build_dispatch_prompt(state, strategic_experience=strategic_experience)
        response_text = ollama.generate(prompt)
        try:
            return parse_llm_response(response_text)
        except:
            # 备选方案：如果 LLM 出错，降级到 SPT
            return PDR.get_dispatch_action(state, rule="SPT")

    print("Running simulation guided by LLM...")
    llm_final_state = Simulator.run_simulation(initial_state, llm_policy)
    llm_metrics = Evaluator.evaluate(llm_final_state)
    
    print("\n=== Final Results ===")
    for t in trajectories:
        print(f"Rule {t['rule']}: Makespan = {t['metrics']['makespan']}")
    print(f"LLM + Reflection: Makespan = {llm_metrics['makespan']}")

if __name__ == "__main__":
    run_baseline_experiment()
