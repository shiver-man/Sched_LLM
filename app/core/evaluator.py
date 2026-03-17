from typing import List, Dict, Any

class Evaluator:
    """
    评估器：负责计算调度轨迹的性能指标。
    """

    @staticmethod
    def evaluate(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算 Makespan, 机器利用率等指标。
        """
        history = state.get("history", [])
        if not history:
            return {"makespan": 0, "utilization": 0}

        # 1. 计算 Makespan (总完工时间)
        makespan = max(event["finish_time"] for event in history if "finish_time" in event)
        
        # 2. 计算机器利用率
        total_process_time = sum(
            event["finish_time"] - event["start_time"] 
            for event in history if "finish_time" in event and "start_time" in event
        )
        num_machines = len(state["machines"])
        # 利用率 = 总加工时间 / (机器数 * 总时间)
        utilization = total_process_time / (num_machines * makespan) if makespan > 0 else 0
        
        # 3. 计算延迟时间 (Tardiness)
        total_tardiness = 0
        job_finish_times = {}
        for event in history:
            job_id = event["job_id"]
            finish_time = event["finish_time"]
            if job_id not in job_finish_times or finish_time > job_finish_times[job_id]:
                job_finish_times[job_id] = finish_time
        
        for job in state["jobs"]:
            due_time = job.get("due_time", float('inf'))
            actual_finish = job_finish_times.get(job["job_id"], 0)
            if actual_finish > due_time:
                total_tardiness += (actual_finish - due_time)

        return {
            "makespan": round(makespan, 2),
            "utilization": round(utilization, 4),
            "total_tardiness": round(total_tardiness, 2),
            "num_events": len(history)
        }
