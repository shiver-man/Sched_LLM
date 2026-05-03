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
                "machine_total_idle_time": 0,
                "vehicle_utilization": 0,
                "transport_wait_time": 0,
                "busiest_vehicle": None,
                "busiest_path": None,
                "path_conflicts": 0,
                "machine_idle_reasons": {"waiting_transport": 0, "waiting_machine": 0, "waiting_repair": 0},
                "machine_idle_reason_durations": {"waiting_transport": 0.0, "waiting_machine": 0.0, "waiting_repair": 0.0},
            }

        makespan = max(event["finish_time"] for event in history)
        total_process_time = sum(event["finish_time"] - event["start_time"] for event in history)
        total_transport_time = sum(event.get("transport_time", 0) for event in history)
        num_machines = max(len(state.get("machines", [])), 1)
        utilization = total_process_time / (num_machines * makespan) if makespan > 0 else 0
        machine_total_idle_time = max(0.0, (num_machines * makespan) - total_process_time)

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
        
        # 重新从 history 中累加运输统计数据，避免深拷贝或状态丢失的问题
        calc_transport_wait_time = 0.0
        calc_vehicle_busy_time = {}
        for h in history:
            if h.get("vehicle_id"):
                v_id = h["vehicle_id"]
                calc_vehicle_busy_time[v_id] = calc_vehicle_busy_time.get(v_id, 0.0) + h.get("transport_time", 0.0)
                # 计算运输等待时间 = 开始运输的时间 - 当前操作派发的时间
                calc_transport_wait_time += max(0.0, h.get("transport_start", 0.0) - h.get("time", 0.0))

        num_vehicles = max(1, len(state.get("vehicles", [])))
        total_vehicle_busy = sum(calc_vehicle_busy_time.values())
        
        # 修正 Vehicle Utilization 的计算逻辑：这里使用所有车辆的忙碌时间除以总的系统可用车时
        vehicle_utilization = total_vehicle_busy / (num_vehicles * makespan) if makespan > 0 else 0
        
        busiest_vehicle = None
        if calc_vehicle_busy_time:
            busiest_vehicle = max(calc_vehicle_busy_time.items(), key=lambda x: x[1])[0]
            
        transport_stats = state.get("transport_stats", {})
        path_loads = transport_stats.get("path_loads", {})
        busiest_path = None
        if path_loads:
            busiest_path = max(path_loads.items(), key=lambda x: x[1])[0]
        idle_reasons = state.get("idle_reason_stats", {"waiting_transport": 0, "waiting_machine": 0, "waiting_repair": 0})
        idle_reason_durations = state.get(
            "idle_reason_durations",
            {"waiting_transport": 0.0, "waiting_machine": 0.0, "waiting_repair": 0.0},
        )

        return {
            "makespan": round(makespan, 4),
            "utilization": round(utilization, 4),
            "total_tardiness": round(total_tardiness, 4),
            "total_transport_time": round(total_transport_time, 4),
            "num_events": num_events,
            "is_complete": is_complete,
            "total_ops_expected": total_ops,
            "machine_total_idle_time": round(machine_total_idle_time, 4),
            "vehicle_utilization": round(vehicle_utilization, 4),
            "transport_wait_time": round(calc_transport_wait_time, 4),
            "busiest_vehicle": busiest_vehicle,
            "busiest_path": busiest_path,
            "path_conflicts": int(transport_stats.get("path_conflicts", 0)),
            "machine_idle_reasons": idle_reasons,
            "machine_idle_reason_durations": {
                "waiting_transport": round(float(idle_reason_durations.get("waiting_transport", 0.0)), 4),
                "waiting_machine": round(float(idle_reason_durations.get("waiting_machine", 0.0)), 4),
                "waiting_repair": round(float(idle_reason_durations.get("waiting_repair", 0.0)), 4),
            },
        }
