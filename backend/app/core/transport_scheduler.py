from typing import Dict, Any, Optional, List
import networkx as nx


class TransportScheduler:
    def __init__(self, state: Dict[str, Any], strategy: str = "NEAREST_VEHICLE"):
        self.state = state
        self.strategy = str(strategy or "NEAREST_VEHICLE").upper()
        self.state.setdefault("transport_stats", {})
        self.state["transport_stats"].setdefault("total_wait_time", 0.0)
        self.state["transport_stats"].setdefault("path_loads", {})
        self.state["transport_stats"].setdefault("vehicle_busy_time", {})
        self.state.setdefault("transport_queue", [])
        self.state["transport_stats"].setdefault("path_reservations", {})
        self.state["transport_stats"].setdefault("path_conflicts", 0)
        for v in self.state.get("vehicles", []):
            v.setdefault("task_queue", [])

    def _travel(self, vehicle: Dict[str, Any], from_loc: str, to_loc: str) -> float:
        if from_loc == to_loc:
            return 0.0
        graph = self.state["graph"]
        carrying = nx.shortest_path_length(graph, from_loc, to_loc, weight="weight") / vehicle["speed"]
        reposition = 0.0
        if vehicle["current_location"] != from_loc:
            reposition = (
                nx.shortest_path_length(graph, vehicle["current_location"], from_loc, weight="weight")
                / vehicle["speed"]
            )
        return reposition + carrying + vehicle.get("load_unload_time", 0.0)

    def _idle_vehicles(self) -> List[Dict[str, Any]]:
        return [v for v in self.state.get("vehicles", []) if v.get("status") == "idle"]

    def _vehicle_score(self, vehicle: Dict[str, Any], from_loc: str, to_loc: str) -> float:
        now = float(self.state.get("time", 0.0))
        travel = self._travel(vehicle, from_loc, to_loc)
        pending = max(0.0, float(vehicle.get("available_time", now)) - now)
        busy = self.state["transport_stats"]["vehicle_busy_time"].get(vehicle["vehicle_id"], 0.0)
        congestion = self._path_congestion_penalty(from_loc, to_loc, now)
        if self.strategy == "LOAD_BALANCING":
            return pending + travel + 0.2 * busy + congestion
        if self.strategy == "SHORTEST_PATH":
            graph = self.state["graph"]
            return float(nx.shortest_path_length(graph, from_loc, to_loc, weight="weight")) + congestion
        return travel + congestion

    def _path_edges(self, from_loc: str, to_loc: str) -> List[str]:
        if from_loc == to_loc:
            return []
        graph = self.state["graph"]
        nodes = nx.shortest_path(graph, from_loc, to_loc, weight="weight")
        return [f"{nodes[i]}->{nodes[i+1]}" for i in range(len(nodes) - 1)]

    def _path_congestion_penalty(self, from_loc: str, to_loc: str, now: float) -> float:
        penalty = 0.0
        reservations = self.state["transport_stats"].get("path_reservations", {})
        for edge in self._path_edges(from_loc, to_loc):
            reserved_until = float(reservations.get(edge, 0.0))
            if reserved_until > now:
                penalty += (reserved_until - now)
        return penalty

    def assign_vehicle(self, job_id: str, op_id: str, from_loc: str, to_loc: str) -> Optional[str]:
        vehicles = self._idle_vehicles()
        if not vehicles:
            return None
        chosen = min(vehicles, key=lambda v: self._vehicle_score(v, from_loc, to_loc))
        task = {"job_id": job_id, "op_id": op_id, "from": from_loc, "to": to_loc}
        chosen["task_queue"].append(task)
        return chosen["vehicle_id"]

    def release_vehicle_task(self, vehicle_id: str, transport_time: float, from_loc: str, to_loc: str) -> None:
        v = next((x for x in self.state.get("vehicles", []) if x["vehicle_id"] == vehicle_id), None)
        if not v:
            return
        if v.get("task_queue"):
            v["task_queue"].pop(0)
        stats = self.state["transport_stats"]
        stats["vehicle_busy_time"][vehicle_id] = stats["vehicle_busy_time"].get(vehicle_id, 0.0) + float(transport_time)
        key = f"{from_loc}->{to_loc}"
        stats["path_loads"][key] = stats["path_loads"].get(key, 0) + 1
        self._release_path_reservations(from_loc, to_loc)

    def reserve_path(self, from_loc: str, to_loc: str, finish_time: float) -> None:
        now = float(self.state.get("time", 0.0))
        reservations = self.state["transport_stats"].setdefault("path_reservations", {})
        conflict = False
        for edge in self._path_edges(from_loc, to_loc):
            if float(reservations.get(edge, 0.0)) > now:
                conflict = True
            reservations[edge] = max(float(reservations.get(edge, 0.0)), float(finish_time))
        if conflict:
            self.state["transport_stats"]["path_conflicts"] = self.state["transport_stats"].get("path_conflicts", 0) + 1

    def _release_path_reservations(self, from_loc: str, to_loc: str) -> None:
        now = float(self.state.get("time", 0.0))
        reservations = self.state["transport_stats"].setdefault("path_reservations", {})
        for edge in self._path_edges(from_loc, to_loc):
            if float(reservations.get(edge, 0.0)) <= now:
                reservations.pop(edge, None)

    def enqueue_transport(self, decision: Dict[str, Any]) -> None:
        key = (decision.get("job_id"), decision.get("op_id"), decision.get("machine_id"))
        for item in self.state["transport_queue"]:
            existing = (item.get("job_id"), item.get("op_id"), item.get("machine_id"))
            if existing == key:
                return
        self.state["transport_queue"].append(
            {
                "job_id": decision.get("job_id"),
                "op_id": decision.get("op_id"),
                "machine_id": decision.get("machine_id"),
                "enqueued_time": float(self.state.get("time", 0.0)),
            }
        )

    def pop_assignable_queue(self) -> Optional[Dict[str, Any]]:
        if not self.state.get("transport_queue"):
            return None
        return self.state["transport_queue"].pop(0)
