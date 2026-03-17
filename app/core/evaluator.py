from typing import Dict, Any

class Evaluator:
    @staticmethod
    def evaluate(state: Dict[str, Any]) -> Dict[str, Any]:
        history = state.get("history", [])
        if not history:
            return {
                "makespan": 0,
                "utilization": 0,
                "total_tardiness": 0,
                "total_transport_time": 0,
                "num_events": 0,
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

        return {
            "makespan": round(makespan, 4),
            "utilization": round(utilization, 4),
            "total_tardiness": round(total_tardiness, 4),
            "total_transport_time": round(total_transport_time, 4),
            "num_events": len(history),
        }