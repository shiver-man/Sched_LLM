from typing import Dict, Any, Optional
import networkx as nx

class Dispatcher:
    """
    调度执行器：负责将 LLM 的调度决策应用到系统状态中。
    """
    
    @staticmethod
    def apply_decision(state: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用单次调度决策。
        
        decision 格式示例:
        {
          "job_id": "J1",
          "op_id": "O11",
          "machine_id": "M1",
          "vehicle_id": "V1",
          "reason": "..."
        }
        """
        job_id = decision.get("job_id")
        machine_id = decision.get("machine_id")
        vehicle_id = decision.get("vehicle_id")
        
        # 1. 找到对应的工件、机器和运输车对象
        job = next((j for j in state["jobs"] if j["job_id"] == job_id), None)
        machine = next((m for m in state["machines"] if m["machine_id"] == machine_id), None)
        vehicle = next((v for v in state["vehicles"] if v["vehicle_id"] == vehicle_id), None) if vehicle_id else None
        
        if not job or not machine:
            return state # 或者抛出异常
            
        # 2. 获取当前工序信息
        current_op = job["operations"][job["current_op_index"]]
        process_time = next(
            (cm["process_time"] for cm in current_op["candidate_machines"] if cm["machine_id"] == machine_id), 
            10.0 # 默认值
        )
        
        # 3. 计算时间（简化版）
        current_time = state["time"]
        
        # 如果需要运输
        transport_time = 0.0
        if vehicle:
            # 简单的距离计算（如果图中有权重）
            try:
                distance = nx.shortest_path_length(state["graph"], vehicle["current_location"], machine["location"], weight='weight')
                transport_time = distance / vehicle["speed"]
            except:
                transport_time = 5.0 # 默认运输时间
            
            # 更新运输车状态
            vehicle["status"] = "busy"
            vehicle["current_location"] = machine["location"]
            vehicle["available_time"] = current_time + transport_time
            
        # 4. 更新机器和工件状态
        start_time = max(current_time + transport_time, job["release_time"], machine["available_time"])
        finish_time = start_time + process_time
        
        machine["status"] = "busy"
        machine["available_time"] = finish_time
        machine["current_job"] = job_id
        
        # 工件进阶到下一工序
        job["current_op_index"] += 1
        if job["current_op_index"] >= len(job["operations"]):
            job["finished"] = True
            
        # 5. 记录历史
        state["history"].append({
            "time": current_time,
            "event": "dispatch",
            "job_id": job_id,
            "op_id": current_op["op_id"],
            "machine_id": machine_id,
            "vehicle_id": vehicle_id,
            "start_time": start_time,
            "finish_time": finish_time,
            "reason": decision.get("reason", "")
        })
        
        # 更新全局系统时间（推移到决策执行开始或结束，视具体逻辑而定）
        # 这里为了演示，假设系统时间推进到该决策产生的完工时间的一小部分，或者保持不变由仿真引擎控制
        # state["time"] = start_time 
        
        return state
