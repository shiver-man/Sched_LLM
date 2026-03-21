from typing import Dict, Any

class Evaluator:
    @staticmethod
    def evaluate(state: Dict[str, Any]) -> Dict[str, Any]:
        history = state.get("history", [])
        if not history:
            total_ops = sum(len(job.get("operations", [])) for job in state.get("jobs", []))
            return {
                "makespan": 0,
                "utilization": 0,
                "total_tardiness": 0,
                "total_transport_time": 0,
                "num_events": 0,
                "is_complete": False,
                "total_ops_expected": total_ops,
                "vehicle_utilization": 0,
                "transport_wait_time": 0,
                "busiest_vehicle": None,
                "busiest_path": None,
                "path_conflicts": 0,
                "machine_idle_reasons": {"waiting_transport": 0, "waiting_machine": 0, "waiting_repair": 0},
            }

        makespan = max(event["finish_time"] for event in history)
        total_process_time = sum(event["finish_time"] - event["start_time"] for event in history)
        total_transport_time = sum(event.get("transport_time", 0) for event in history)
        num_machines = max(len(state.get("machines", [])), 1)
        utilization = total_process_time / (num_machines * makespan) if makespan > 0 else 0

        job_finish_times = {}
        for event in history:
            job_id = event["job_id"]
            finish_time = event["finish_time"]
            if job_id not in job_finish_times or finish_time > job_finish_times[job_id]:
                job_finish_times[job_id] = finish_time

        total_tardiness = 0.0
        for job in state["jobs"]:
            due_time = job.get("due_time", float("inf"))
            actual_finish = job_finish_times.get(job["job_id"], 0)
            if actual_finish > due_time:
                total_tardiness += actual_finish - due_time

        # 计算预期总工序数
        # 注意：如果仿真中动态新增了任务，这里的总数应包含这些任务
        total_ops = sum(len(job["operations"]) for job in state["jobs"])
        num_events = len(history)
        
        # 统计 history 中实际完成的唯一工序数，防止重复派工干扰
        finished_ops = set()
        for h in history:
            finished_ops.add((h["job_id"], h["op_id"]))
        
        is_complete = (len(finished_ops) >= total_ops) and (total_ops > 0)
        transport_stats = state.get("transport_stats", {})
        vehicle_busy_map = transport_stats.get("vehicle_busy_time", {})
        num_vehicles = max(1, len(state.get("vehicles", [])))
        total_vehicle_busy = sum(vehicle_busy_map.values())
        vehicle_utilization = total_vehicle_busy / (num_vehicles * makespan) if makespan > 0 else 0
        busiest_vehicle = None
        if vehicle_busy_map:
            busiest_vehicle = max(vehicle_busy_map.items(), key=lambda x: x[1])[0]
        path_loads = transport_stats.get("path_loads", {})
        busiest_path = None
        if path_loads:
            busiest_path = max(path_loads.items(), key=lambda x: x[1])[0]
        idle_reasons = state.get("idle_reason_stats", {"waiting_transport": 0, "waiting_machine": 0, "waiting_repair": 0})

        return {
            "makespan": round(makespan, 4),
            "utilization": round(utilization, 4),
            "total_tardiness": round(total_tardiness, 4),
            "total_transport_time": round(total_transport_time, 4),
            "num_events": num_events,
            "is_complete": is_complete,
            "total_ops_expected": total_ops,
            "vehicle_utilization": round(vehicle_utilization, 4),
            "transport_wait_time": round(float(transport_stats.get("total_wait_time", 0.0)), 4),
            "busiest_vehicle": busiest_vehicle,
            "busiest_path": busiest_path,
            "path_conflicts": int(transport_stats.get("path_conflicts", 0)),
            "machine_idle_reasons": idle_reasons,
        }
