from typing import List, Dict, Any, Optional

class PDR:
    """
    Priority Dispatching Rules (PDRs) 启发式调度规则。
    用于生成基准调度轨迹 (Trajectories)。
    """

    @staticmethod
    def get_dispatch_action(state: Dict[str, Any], rule: str = "SPT") -> Optional[Dict[str, Any]]:
        """
        根据指定的启发式规则返回下一步调度动作。
        """
        # 1. 筛选当前可执行的工序（工件已释放且未完成）
        dispatchable_jobs = [
            j for j in state["jobs"] 
            if not j["finished"] and state["time"] >= j["release_time"]
        ]
        
        if not dispatchable_jobs:
            return None
            
        # 2. 筛选当前空闲的机器
        available_machines = [
            m for m in state["machines"] if m["status"] == "idle"
        ]
        
        if not available_machines:
            return None

        # 3. 收集所有可能的 (Job, Op, Machine) 组合
        candidates = []
        for job in dispatchable_jobs:
            current_op = job["operations"][job["current_op_index"]]
            for machine in available_machines:
                # 检查该机器是否在候选机器列表中
                cm = next((m for m in current_op["candidate_machines"] if m["machine_id"] == machine["machine_id"]), None)
                if cm:
                    candidates.append({
                        "job": job,
                        "op": current_op,
                        "machine": machine,
                        "process_time": cm["process_time"]
                    })
        
        if not candidates:
            return None

        # 4. 根据规则进行排序
        if rule == "SPT": # Shortest Processing Time
            selected = min(candidates, key=lambda x: x["process_time"])
        elif rule == "FIFO": # First In First Out
            selected = min(candidates, key=lambda x: x["job"]["release_time"])
        elif rule == "MWKR": # Most Work Remaining
            selected = max(candidates, key=lambda x: sum(
                max(cm["process_time"] for cm in op["candidate_machines"])
                for op in x["job"]["operations"][x["job"]["current_op_index"]:]
            ))
        else: # 默认随机选择第一个
            selected = candidates[0]

        # 5. 选择运输车（如果有）
        available_vehicles = [v for v in state["vehicles"] if v["status"] == "idle"]
        vehicle_id = available_vehicles[0]["vehicle_id"] if available_vehicles else None

        return {
            "job_id": selected["job"]["job_id"],
            "op_id": selected["op"]["op_id"],
            "machine_id": selected["machine"]["machine_id"],
            "vehicle_id": vehicle_id,
            "reason": f"Heuristic Rule: {rule}"
        }
