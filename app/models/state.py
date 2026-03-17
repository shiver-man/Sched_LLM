from typing import Dict, Any
import networkx as nx
from app.models.schema import ScheduleRequest


def build_layout_graph(layout) -> nx.Graph:
    graph = nx.Graph()
    for node in layout.nodes:
        graph.add_node(node.node_id)
    for edge in layout.edges:
        graph.add_edge(edge.from_node, edge.to_node, weight=edge.distance)
    return graph


def build_initial_state(req: ScheduleRequest) -> Dict[str, Any]:
    graph = build_layout_graph(req.layout)

    jobs_state = []
    for job in req.jobs:
        jobs_state.append({
            "job_id": job.job_id,
            "release_time": job.release_time,
            "due_time": job.due_time,
            "current_op_index": 0,
            "finished": False,
            "operations": [
                {
                    "op_id": op.op_id,
                    "candidate_machines": [
                        {
                            "machine_id": cm.machine_id,
                            "process_time": cm.process_time
                        }
                        for cm in op.candidate_machines
                    ]
                }
                for op in job.operations
            ]
        })

    machines_state = []
    for machine in req.machines:
        machines_state.append({
            "machine_id": machine.machine_id,
            "machine_type": machine.machine_type,
            "location": machine.location,
            "status": "idle",
            "available_time": req.current_time,
            "current_job": None
        })

    vehicles_state = []
    for vehicle in req.vehicles:
        vehicles_state.append({
            "vehicle_id": vehicle.vehicle_id,
            "current_location": vehicle.current_location,
            "speed": vehicle.speed,
            "capacity": vehicle.capacity,
            "status": "idle",
            "available_time": req.current_time,
            "current_task": None
        })

    state = {
        "time": req.current_time,
        "jobs": jobs_state,
        "machines": machines_state,
        "vehicles": vehicles_state,
        "objective": req.objective.model_dump(),
        "graph": graph,
        "dispatchable_ops": [],
        "transport_tasks": [],
        "history": []
    }

    return state