from typing import Dict, Any, Optional
import networkx as nx


class Dispatcher:
    @staticmethod
    def _normalize_id(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().strip("\"'`")
        text = text.strip("，,。.;；:：")
        return text.casefold()

    @staticmethod
    def _find_job(state: Dict[str, Any], job_id: str) -> Optional[Dict[str, Any]]:
        if job_id is None:
            return None
        direct = next((j for j in state["jobs"] if j["job_id"] == job_id), None)
        if direct is not None:
            return direct
        normalized = Dispatcher._normalize_id(job_id)
        return next(
            (j for j in state["jobs"] if Dispatcher._normalize_id(j["job_id"]) == normalized),
            None,
        )

    @staticmethod
    def _find_machine(state: Dict[str, Any], machine_id: str) -> Optional[Dict[str, Any]]:
        if machine_id is None:
            return None
        direct = next((m for m in state["machines"] if m["machine_id"] == machine_id), None)
        if direct is not None:
            return direct
        normalized = Dispatcher._normalize_id(machine_id)
        return next(
            (m for m in state["machines"] if Dispatcher._normalize_id(m["machine_id"]) == normalized),
            None,
        )

    @staticmethod
    def _find_vehicle(state: Dict[str, Any], vehicle_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not vehicle_id:
            return None
        direct = next((v for v in state["vehicles"] if v["vehicle_id"] == vehicle_id), None)
        if direct is not None:
            return direct
        normalized = Dispatcher._normalize_id(vehicle_id)
        return next(
            (v for v in state["vehicles"] if Dispatcher._normalize_id(v["vehicle_id"]) == normalized),
            None,
        )

    @staticmethod
    def validate_decision(state: Dict[str, Any], decision: Dict[str, Any]) -> None:
        job = Dispatcher._find_job(state, decision.get("job_id"))
        machine = Dispatcher._find_machine(state, decision.get("machine_id"))
        vehicle = Dispatcher._find_vehicle(state, decision.get("vehicle_id"))

        if job is None:
            raise ValueError("job_id 不存在")
        if machine is None:
            raise ValueError("machine_id 不存在")
        if job["finished"]:
            raise ValueError("该工件已经完成")
        if state["time"] < job["release_time"]:
            raise ValueError("该工件尚未释放")
        if state["time"] < job.get("ready_time", job["release_time"]):
            raise ValueError("该工件尚未准备好")
        if machine["status"] != "idle":
            raise ValueError("目标机器当前非空闲")

        current_op = job["operations"][job["current_op_index"]]
        cm = next(
            (x for x in current_op["candidate_machines"] if x["machine_id"] == machine["machine_id"]),
            None,
        )
        if cm is None:
            raise ValueError("目标机器不在当前工序的候选机器列表中")

        if job["current_location"] != machine["location"]:
            if vehicle is None:
                raise ValueError("需要运输，但 vehicle_id 为空")
            if vehicle["status"] != "idle":
                raise ValueError("目标车辆当前非空闲")

    @staticmethod
    def _calc_transport(state: Dict[str, Any], from_loc: str, to_loc: str, vehicle: Dict[str, Any]):
        if from_loc == to_loc:
            return 0.0, 0.0, 0.0

        graph = state["graph"]

        reposition_dist = 0.0
        if vehicle["current_location"] != from_loc:
            reposition_dist = nx.shortest_path_length(
                graph,
                vehicle["current_location"],
                from_loc,
                weight="weight",
            )

        carrying_dist = nx.shortest_path_length(
            graph,
            from_loc,
            to_loc,
            weight="weight",
        )

        reposition_time = reposition_dist / vehicle["speed"]
        carrying_time = carrying_dist / vehicle["speed"]
        load_unload_time = vehicle.get("load_unload_time", 0.0)
        total_time = reposition_time + carrying_time + load_unload_time

        return reposition_time, carrying_time, total_time

    @staticmethod
    def apply_decision(state: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        Dispatcher.validate_decision(state, decision)

        current_time = state["time"]
        job = Dispatcher._find_job(state, decision["job_id"])
        machine = Dispatcher._find_machine(state, decision["machine_id"])
        vehicle = Dispatcher._find_vehicle(state, decision.get("vehicle_id"))

        current_op = job["operations"][job["current_op_index"]]
        cm = next(
            x for x in current_op["candidate_machines"]
            if x["machine_id"] == machine["machine_id"]
        )
        process_time = cm["process_time"]

        transport_time = 0.0
        reposition_time = 0.0
        carrying_time = 0.0
        transport_start = current_time
        transport_finish = current_time

        if job["current_location"] != machine["location"]:
            reposition_time, carrying_time, transport_time = Dispatcher._calc_transport(
                state,
                job["current_location"],
                machine["location"],
                vehicle,
            )
            transport_start = max(current_time, vehicle["available_time"])
            transport_finish = transport_start + transport_time

            vehicle["status"] = "busy"
            vehicle["available_time"] = transport_finish
            vehicle["current_location"] = machine["location"]
            vehicle["current_task"] = f"{job['job_id']}->{machine['machine_id']}"

            job["current_location"] = machine["location"]

        start_time = max(
            current_time,
            transport_finish,
            machine["available_time"],
            job.get("ready_time", job["release_time"]),
        )
        finish_time = start_time + process_time

        machine["status"] = "busy"
        machine["available_time"] = finish_time
        machine["current_job"] = job["job_id"]

        job["current_op_index"] += 1
        job["ready_time"] = finish_time

        if job["current_op_index"] >= len(job["operations"]):
            job["finished"] = True

        state["history"].append(
            {
                "time": current_time,
                "event": "dispatch",
                "job_id": job["job_id"],
                "op_id": current_op["op_id"],
                "machine_id": machine["machine_id"],
                "vehicle_id": vehicle["vehicle_id"] if vehicle else None,
                "transport_start": transport_start,
                "transport_finish": transport_finish,
                "reposition_time": reposition_time,
                "carrying_time": carrying_time,
                "transport_time": transport_time,
                "start_time": start_time,
                "finish_time": finish_time,
                "reason": decision.get("reason", ""),
            }
        )

        return state
