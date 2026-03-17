from typing import Dict, Any, Optional, List
import networkx as nx
from app.models.state import get_dispatchable_jobs

class PDR:
    """Priority Dispatching Rules 基准调度器。"""

    @staticmethod
    def _transport_time(
        state: Dict[str, Any],
        from_loc: str,
        to_loc: str,
        vehicle: Optional[Dict[str, Any]],
    ) -> float:
        if from_loc == to_loc:
            return 0.0
        if vehicle is None:
            return float("inf")
        graph = state["graph"]
        distance = nx.shortest_path_length(graph, from_loc, to_loc, weight="weight")
        reposition = 0.0
        if vehicle["current_location"] != from_loc:
            reposition = nx.shortest_path_length(
                graph,
                vehicle["current_location"],
                from_loc,
                weight="weight",
            ) / vehicle["speed"]
        carrying = distance / vehicle["speed"]
        return reposition + carrying + vehicle.get("load_unload_time", 0.0)

    @staticmethod
    def get_dispatch_action(state: Dict[str, Any], rule: str = "SPT") -> Optional[Dict[str, Any]]:
        dispatchable_jobs = get_dispatchable_jobs(state)
        if not dispatchable_jobs:
            return None

        available_machines = [m for m in state["machines"] if m["status"] == "idle"]
        if not available_machines:
            return None

        idle_vehicles = [v for v in state["vehicles"] if v["status"] == "idle"]
        candidates: List[Dict[str, Any]] = []

        for job in dispatchable_jobs:
            current_op = job["operations"][job["current_op_index"]]
            from_loc = job["current_location"]
            for machine in available_machines:
                cm = next(
                    (x for x in current_op["candidate_machines"] if x["machine_id"] == machine["machine_id"]),
                    None,
                )
                if not cm:
                    continue

                if from_loc == machine["location"]:
                    candidates.append(
                        {
                            "job": job,
                            "op": current_op,
                            "machine": machine,
                            "vehicle": None,
                            "process_time": cm["process_time"],
                            "transport_time": 0.0,
                        }
                    )
                else:
                    for vehicle in idle_vehicles:
                        t_time = PDR._transport_time(state, from_loc, machine["location"], vehicle)
                        candidates.append(
                            {
                                "job": job,
                                "op": current_op,
                                "machine": machine,
                                "vehicle": vehicle,
                                "process_time": cm["process_time"],
                                "transport_time": t_time,
                            }
                        )
            if not candidates:
                return None

            if rule == "SPT":
                selected = min(candidates, key=lambda x: x["process_time"] + x["transport_time"])
            elif rule == "FIFO":
                selected = min(candidates, key=lambda x: x["job"]["release_time"])
            elif rule == "MWKR":
                def remaining_work(item):
                    job = item["job"]
                    rest_ops = job["operations"][job["current_op_index"]:]
                    return sum(
                        min(cm["process_time"] for cm in op["candidate_machines"])
                        for op in rest_ops
                    )

                selected = max(candidates, key=remaining_work)
            else:
                selected = candidates[0]

            return {
                "job_id": selected["job"]["job_id"],
                "op_id": selected["op"]["op_id"],
                "machine_id": selected["machine"]["machine_id"],
                "vehicle_id": selected["vehicle"]["vehicle_id"] if selected["vehicle"] else None,
                "reason": f"Heuristic Rule: {rule}",
            }