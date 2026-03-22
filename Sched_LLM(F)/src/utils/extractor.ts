import type { Job, Machine, ScheduleBaseData, TransportEdge, Vehicle } from "../types";

function toSingleLine(input: string): string {
  return input.replace(/\s+/g, " ").trim();
}

function pickNumber(text: string, pattern: RegExp): number | undefined {
  const match = text.match(pattern);
  if (!match?.[1]) {
    return undefined;
  }
  const value = Number(match[1]);
  return Number.isFinite(value) ? value : undefined;
}

function pickWorkshop(text: string): string | undefined {
  const match = text.match(/(?:车间|工厂|产线)\s*([A-Za-z0-9_\-\u4e00-\u9fa5]+?)(?=(?:有|,|，|。|\s|$))/i);
  return match?.[1];
}

function pickAlgorithm(text: string): "GA" | "PPO" | undefined {
  if (/(^|[\s，。,:：])ppo([\s，。,:：]|$)/i.test(text)) {
    return "PPO";
  }
  if (/(^|[\s，。,:：])ga([\s，。,:：]|$)/i.test(text)) {
    return "GA";
  }
  return undefined;
}

function pickFactoryName(text: string): string {
  const match = text.match(/(?:工厂|factory)\s*([A-Za-z0-9_\-\u4e00-\u9fa5]+)/i);
  return match?.[1] ? `工厂${match[1]}` : "默认工厂";
}

function buildMachines(machineCount: number): Machine[] {
  return Array.from({ length: machineCount }, (_, index) => {
    const machineIndex = index + 1;
    return {
      machine_id: `M${machineIndex}`,
      location: `L${machineIndex}`,
      status: "idle",
      type: machineIndex % 2 === 0 ? "assembly" : "processing"
    };
  });
}

function buildVehicles(vehicleCount: number): Vehicle[] {
  return Array.from({ length: vehicleCount }, (_, index) => ({
    vehicle_id: `V${index + 1}`,
    start_location: "WH",
    speed: 1.2,
    load_unload_time: 8,
    status: "idle"
  }));
}

function buildNodes(machines: Machine[]): string[] {
  const machineNodes = machines.map((machine) => machine.location);
  return ["WH", "BUFFER", ...machineNodes];
}

function buildEdges(nodes: string[]): TransportEdge[] {
  const edges: TransportEdge[] = [];
  for (let index = 0; index < nodes.length - 1; index += 1) {
    edges.push({
      from: nodes[index],
      to: nodes[index + 1],
      distance: 10 + index * 3
    });
  }
  return edges;
}

function buildJobs(jobCount: number, planningHorizon: number, machines: Machine[]): Job[] {
  const candidateMachineCount = Math.min(3, machines.length);
  const candidateMachines = machines.slice(0, candidateMachineCount);
  return Array.from({ length: jobCount }, (_, index) => {
    const jobIndex = index + 1;
    const operationBase = 20 + index * 2;
    return {
      job_id: `J${jobIndex}`,
      release_time: Math.max(0, index * 5),
      due_time: planningHorizon - (jobCount - jobIndex) * 3,
      initial_location: "WH",
      operations: [
        {
          operation_id: `J${jobIndex}-OP1`,
          candidate_machines: candidateMachines.map((machine, machineIndex) => ({
            machine_id: machine.machine_id,
            processing_time: operationBase + machineIndex * 3
          }))
        },
        {
          operation_id: `J${jobIndex}-OP2`,
          candidate_machines: candidateMachines.map((machine, machineIndex) => ({
            machine_id: machine.machine_id,
            processing_time: operationBase + 12 + machineIndex * 4
          }))
        }
      ]
    };
  });
}

export function extractScheduleBaseData(input: string): ScheduleBaseData {
  const normalized = toSingleLine(input);
  const machineCount = pickNumber(normalized, /(\d+(?:\.\d+)?)\s*(?:台设备|台机器|台机床|machines?)/i) ?? 6;
  const agvCount = pickNumber(normalized, /(\d+(?:\.\d+)?)\s*(?:辆\s*agv|台\s*agv|agv)/i) ?? 3;
  const orderCount = pickNumber(normalized, /(\d+(?:\.\d+)?)\s*(?:个订单|单任务|jobs?|orders?)/i) ?? 10;
  const shiftHours = pickNumber(normalized, /(\d+(?:\.\d+)?)\s*(?:小时|h|hours?)/i) ?? 8;
  const planningHorizon = Math.max(1, Math.round(shiftHours * 60));
  const machines = buildMachines(Math.max(1, Math.round(machineCount)));
  const vehicles = buildVehicles(Math.max(1, Math.round(agvCount)));
  const nodes = buildNodes(machines);
  const jobs = buildJobs(Math.max(1, Math.round(orderCount)), planningHorizon, machines);

  return {
    factory_info: {
      factory_id: `factory_${pickWorkshop(normalized) ?? "001"}`,
      factory_name: pickFactoryName(normalized),
      planning_horizon: planningHorizon,
      current_time: new Date().toISOString()
    },
    shop_floor: {
      machines,
      vehicles,
      transport_network: {
        nodes,
        edges: buildEdges(nodes)
      }
    },
    jobs,
    simulation_config: {
      random_seed: 42,
      ppo_max_steps: 2000,
      ppo_episodes: 300,
      ppo_noise_low: 0.01,
      ppo_noise_high: 0.15
    },
    uncertainty_config: {
      processing: {
        low: -0.1,
        high: 0.1
      },
      transport: {
        low: -0.08,
        high: 0.12
      },
      breakdown: {
        probability: 0.02,
        repair_time: 20
      },
      vehicle_delay: {
        probability: 0.05,
        low: 1,
        high: 8
      }
    },
    source_text: input,
    algorithm_preference: pickAlgorithm(normalized)
  };
}
