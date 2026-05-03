import asyncio
import json
from app.api.routes_simulation import run_unified_simulation
from app.models.schema import ScheduleRequest
from app.core.simulator import Simulator

async def main():
    # 从 MATLAB 数据手动构建对应的 JSON Payload 
    # 这些数据源自你提供的 mk01.txt 结尾部分
    
    # 距离/时间矩阵 TRT (LU(0) + M1~M6(1-6))
    TRT = [ 
         [0,  8, 10, 12, 14, 16, 18], 
         [8,  0,  6,  9, 11, 13, 15], 
         [10, 6,  0,  5,  8, 10, 12], 
         [12, 9,  5,  0,  6,  8, 10], 
         [14, 11, 8,  6,  0,  5,  7], 
         [16, 13, 10, 8,  5,  0,  4], 
         [18, 15, 12, 10, 7,  4,  0]
    ]
    nodes = ["LU", "M1", "M2", "M3", "M4", "M5", "M6"]
    edges = []
    for i in range(7):
        for j in range(7):
            if i != j:
                edges.append({"from": nodes[i], "to": nodes[j], "distance": float(TRT[i][j])})

    # 工件释放和交期时间
    Rel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Due = [90, 85, 88, 92, 96, 100, 94, 98, 102, 105]
    Jpos_idx = [0, 6, 5, 4, 3, 2, 1, 6, 4, 2] # MATLAB索引转节点

    # 解析标准的 MK01 工序数据（此处硬编码简化，对应你 mk01.fjs 的结构）
    from experiments.fjsp_case_runner import parse_fjs
    from pathlib import Path
    base_payload = parse_fjs(Path("FJSP_epl/FJSP算例/Monaldo/Fjsp/Job_Data/Brandimarte_Data/Text/Mk01.fjs"))
    
    jobs = base_payload["jobs"]
    for i, j in enumerate(jobs):
        j["release_time"] = float(Rel[i])
        j["due_time"] = float(Due[i])
        j["initial_location"] = nodes[Jpos_idx[i]]
    
    machines = [
        {"machine_id": f"M{i}", "type": "FJSP_MACHINE", "location": f"M{i}", "status": "idle"}
        for i in range(1, 7)
    ]
    
    # 修改为 1 辆车，速度 0.5，装卸时间 3，初始在 M6
    vehicles = [
        {"vehicle_id": "V1", "start_location": "M6", "speed": 0.5, "load_unload_time": 3.0, "status": "idle"}
    ]

    payload = {
        "factory_info": {"factory_id": "Custom_MK01", "factory_name": "Custom Bench", "planning_horizon": 50000.0, "current_time": 0.0},
        "shop_floor": {"machines": machines, "vehicles": vehicles, "transport_network": {"nodes": nodes, "edges": edges}},
        "jobs": jobs,
        "selected_strategies": ["COOP_RH"],
        "simulation_config": {"random_seed": 42},
        "dispatching_config": {"transport_rule": "NEAREST_VEHICLE"},
        "llm_config": {"use_ollama": False},
        "return_raw_json": False,
    }

    print("Running Custom MATLAB Config Payload...")
    res = await run_unified_simulation(payload)
    
    # 适配 Pydantic 模型的访问方式
    summary = res.summary_comparison[0]
    if isinstance(summary, dict):
        print(f"Makespan: {summary['makespan']}")
        print(f"Transport Time: {summary['transport_time']}")
        print(f"Vehicle Utilization: {summary['vehicle_utilization']:.2f}")
        print(f"Transport Wait Time: {summary['transport_wait_time']:.2f}")
    else:
        print(f"Makespan: {summary.makespan}")
        print(f"Transport Time: {summary.transport_time}")
        print(f"Vehicle Utilization: {summary.vehicle_utilization:.2f}")
        print(f"Transport Wait Time: {summary.transport_wait_time:.2f}")
    
    # 把图片保存下来看
    if res.gantt_chart_base64:
        import base64
        svg_data = res.gantt_chart_base64.split(",")[1]
        with open("custom_mk01_gantt.svg", "wb") as f:
            f.write(base64.b64decode(svg_data))
        print("Gantt chart saved to custom_mk01_gantt.svg")

if __name__ == "__main__":
    asyncio.run(main())