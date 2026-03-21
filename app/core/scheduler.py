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
    def _transport_metrics(
        state: Dict[str, Any],
        from_loc: str,
        to_loc: str,
        vehicle: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        if from_loc == to_loc:
            return {"total": 0.0, "reposition": 0.0, "carrying": 0.0}
        if vehicle is None:
            return {"total": float("inf"), "reposition": float("inf"), "carrying": float("inf")}
        graph = state["graph"]
        carrying = nx.shortest_path_length(graph, from_loc, to_loc, weight="weight") / vehicle["speed"]
        reposition = 0.0
        if vehicle["current_location"] != from_loc:
            reposition = (
                nx.shortest_path_length(graph, vehicle["current_location"], from_loc, weight="weight")
                / vehicle["speed"]
            )
        total = reposition + carrying + vehicle.get("load_unload_time", 0.0)
        return {"total": total, "reposition": reposition, "carrying": carrying}

    @staticmethod
    def _downstream_risk(state: Dict[str, Any], job: Dict[str, Any], machine: Dict[str, Any]) -> float:
        next_index = job["current_op_index"] + 1
        if next_index >= len(job["operations"]):
            return 0.0
        next_op = job["operations"][next_index]
        graph = state["graph"]
        from_loc = machine["location"]
        candidates = []
        for cm in next_op["candidate_machines"]:
            target_machine = next((m for m in state["machines"] if m["machine_id"] == cm["machine_id"]), None)
            if not target_machine:
                continue
            dist = nx.shortest_path_length(graph, from_loc, target_machine["location"], weight="weight")
            candidates.append(dist)
        if not candidates:
            return 0.0
        return float(min(candidates))

    @staticmethod
    def _rolling_horizon_penalty(
        state: Dict[str, Any],
        machine_id: str,
        projected_finish: float,
        lookahead_window: float = 30.0,
    ) -> float:
        penalty = 0.0
        for j in state["jobs"]:
            if j.get("finished"):
                continue
            idx = j.get("current_op_index", 0)
            if idx >= len(j["operations"]):
                continue
            op = j["operations"][idx]
            release = max(j.get("release_time", 0.0), j.get("ready_time", j.get("release_time", 0.0)))
            if release > projected_finish + lookahead_window:
                continue
            candidates = op.get("candidate_machines", [])
            if len(candidates) == 1 and candidates[0]["machine_id"] == machine_id:
                penalty += 1.0
        machine = next((m for m in state["machines"] if m["machine_id"] == machine_id), None)
        queue = 0.0
        if machine:
            queue = max(0.0, machine.get("available_time", 0.0) - state.get("time", 0.0))
        return penalty + 0.05 * queue

    @staticmethod
    def _rush_bonus(job: Dict[str, Any], now: float) -> float:
        job_id = str(job.get("job_id", "")).upper()
        due_slack = float(job.get("due_time", now) - now)
        urgency = 0.0
        if due_slack < 30:
            urgency += (30 - due_slack) / 30.0
        if "RUSH" in job_id or "URGENT" in job_id:
            urgency += 1.0
        return urgency

    @staticmethod
    def get_dispatch_action(state: Dict[str, Any], rule: str = "SPT") -> Optional[Dict[str, Any]]:
        dispatchable_jobs = [j for j in get_dispatchable_jobs(state) if not j.get("locked", False)]
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
                            "vehicle_wait": 0.0,
                            "downstream_risk": PDR._downstream_risk(state, job, machine),
                            "machine_queue": max(0.0, machine.get("available_time", 0.0) - state.get("time", 0.0)),
                            "empty_run": 0.0,
                            "rh_penalty": PDR._rolling_horizon_penalty(
                                state,
                                machine["machine_id"],
                                state.get("time", 0.0) + cm["process_time"],
                            ),
                            "rush_bonus": PDR._rush_bonus(job, state.get("time", 0.0)),
                        }
                    )
                else:
                    for vehicle in idle_vehicles:
                        tm = PDR._transport_metrics(state, from_loc, machine["location"], vehicle)
                        t_time = tm["total"]
                        candidates.append(
                            {
                                "job": job,
                                "op": current_op,
                                "machine": machine,
                                "vehicle": vehicle,
                                "process_time": cm["process_time"],
                                "transport_time": t_time,
                                "vehicle_wait": max(0.0, vehicle.get("available_time", 0.0) - state.get("time", 0.0)),
                                "downstream_risk": PDR._downstream_risk(state, job, machine),
                                "machine_queue": max(0.0, machine.get("available_time", 0.0) - state.get("time", 0.0)),
                                "empty_run": tm["reposition"],
                                "rh_penalty": PDR._rolling_horizon_penalty(
                                    state,
                                    machine["machine_id"],
                                    state.get("time", 0.0) + t_time + cm["process_time"],
                                ),
                                "rush_bonus": PDR._rush_bonus(job, state.get("time", 0.0)),
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
        elif rule in {"COOP", "COOP_RH"}:
            weights = state.get("metadata", {}).get("dispatching_config", {}).get("joint_score_weights", {})
            alpha = float(weights.get("alpha", 1.0))
            beta = float(weights.get("beta", 1.0))
            gamma = float(weights.get("gamma", 0.6))
            delta = float(weights.get("delta", 0.5))
            epsilon = float(weights.get("epsilon", 0.3))
            eta = float(weights.get("eta", 0.4))
            zeta = float(weights.get("zeta", 0.8 if rule == "COOP_RH" else 0.4))
            kappa = float(weights.get("kappa", 0.9))
            selected = min(
                candidates,
                key=lambda x: (
                    alpha * x["process_time"]
                    + beta * x["transport_time"]
                    + gamma * x["vehicle_wait"]
                    + delta * x["downstream_risk"]
                    + epsilon * x["machine_queue"]
                    + eta * x["empty_run"]
                    + zeta * x["rh_penalty"]
                    - kappa * x["rush_bonus"]
                ),
            )
        else:
            selected = candidates[0]

        return {
            "job_id": selected["job"]["job_id"],
            "op_id": selected["op"]["op_id"],
            "machine_id": selected["machine"]["machine_id"],
            "vehicle_id": selected["vehicle"]["vehicle_id"] if selected["vehicle"] else None,
            "reason": f"Heuristic Rule: {rule}",
        }
