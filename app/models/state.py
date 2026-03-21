from typing import Dict, Any, List
import networkx as nx
from app.models.schema import ScheduleRequest

def build_graph(layout) -> nx.Graph:
    graph = nx.DiGraph() if layout.directed else nx.Graph()
    for node in layout.nodes:
        graph.add_node(normalize_id(node))
    for edge in layout.edges:
        graph.add_edge(normalize_id(edge.from_node), normalize_id(edge.to_node), weight=edge.distance)
    return graph

def normalize_id(value: Any) -> str:
    """统一规范化 ID：去除前后空格、统一小写。"""
    if value is None:
        return ""
    return str(value).strip().casefold()

def build_initial_state(req: ScheduleRequest) -> Dict[str, Any]:
    graph = build_graph(req.layout)

    jobs = []
    for job in req.jobs:
        initial_location = job.initial_location
        if not initial_location and job.operations:
            initial_location = job.operations[0].source_location
        if not initial_location and req.layout.nodes:
            initial_location = req.layout.nodes[0]
        if not initial_location:
            raise ValueError(f"工件 {job.job_id} 缺少 initial_location，且无法从布局推断")

        jobs.append(
            {
                "job_id": normalize_id(job.job_id),
                "operations": [
                    {
                        "op_id": normalize_id(op.op_id),
                        "source_location": normalize_id(op.source_location),
                        "candidate_machines": [
                            {
                                "machine_id": normalize_id(cm.machine_id),
                                "process_time": cm.process_time,
                            }
                            for cm in op.candidate_machines
                        ],
                    }
                    for op in job.operations
                ],
                "release_time": job.release_time,
                "due_time": job.due_time,
                "initial_location": normalize_id(initial_location),
                "current_location": normalize_id(initial_location),
                "current_op_index": 0,
                "finished": False,
                "ready_time": job.release_time,
                "in_transport": False,
                "next_machine_id": None,
            }
        )
    machines = []
    for machine in req.machines:
        machines.append(
            {
                "machine_id": normalize_id(machine.machine_id),
                "machine_type": machine.machine_type,
                "location": normalize_id(machine.location),
                "status": str(machine.status).lower(),
                "available_time": machine.available_time,
                "current_job": normalize_id(machine.current_job) if machine.current_job else None,
            }
        )

    vehicles = []
    for vehicle in req.vehicles:
        vehicles.append(
            {
                "vehicle_id": normalize_id(vehicle.vehicle_id),
                "current_location": normalize_id(vehicle.current_location),
                "speed": vehicle.speed,
                "capacity": vehicle.capacity,
                "load_unload_time": vehicle.load_unload_time,
                "status": str(vehicle.status).lower(),
                "available_time": vehicle.available_time,
                "current_task": vehicle.current_task,
                "task_queue": [],
            }
        )

    return {
        "time": req.current_time,
        "graph": graph,
        "jobs": jobs,
        "machines": machines,
        "vehicles": vehicles,
        "history": [],
        "transport_queue": [],
        "transport_stats": {
            "total_wait_time": 0.0,
            "path_loads": {},
            "vehicle_busy_time": {},
            "path_reservations": {},
            "path_conflicts": 0,
        },
        "idle_reason_stats": {
            "waiting_transport": 0,
            "waiting_machine": 0,
            "waiting_repair": 0,
        },
        "strategic_experience": req.strategic_experience,
        "metadata": req.metadata or {},
    }

def get_dispatchable_jobs(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = []
    for job in state["jobs"]:
        if job["finished"]:
            continue
        if state["time"] < job["release_time"]:
            continue
        if state["time"] < job.get("ready_time", job["release_time"]):
            continue
        result.append(job)
    return result
