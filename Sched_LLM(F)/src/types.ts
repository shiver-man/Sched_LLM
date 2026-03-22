export type Role = "user" | "assistant";

export type MessageContentType = "text" | "image";

export interface ChatMessage {
  id: string;
  role: Role;
  contentType: MessageContentType;
  content: string;
  createdAt: number;
}

export interface ScheduleMetrics {
  makespan?: number;
  vehicleUtilization?: number;
  transportWaitTime?: number;
}

export interface ScheduleResult {
  replyText: string;
  plan?: unknown;
  metrics?: ScheduleMetrics;
  raw?: unknown;
}

export interface FactoryInfo {
  factory_id: string;
  factory_name: string;
  planning_horizon: number;
  current_time: string;
}

export interface Machine {
  machine_id: string;
  location: string;
  status: "idle" | "busy" | "down";
  type: string;
}

export interface Vehicle {
  vehicle_id: string;
  start_location: string;
  speed: number;
  load_unload_time: number;
  status: "idle" | "busy" | "down";
}

export interface TransportEdge {
  from: string;
  to: string;
  distance: number;
}

export interface ShopFloor {
  machines: Machine[];
  vehicles: Vehicle[];
  transport_network: {
    nodes: string[];
    edges: TransportEdge[];
  };
}

export interface CandidateMachine {
  machine_id: string;
  processing_time: number;
}

export interface JobOperation {
  operation_id: string;
  candidate_machines: CandidateMachine[];
}

export interface Job {
  job_id: string;
  release_time: number;
  due_time: number;
  initial_location: string;
  operations: JobOperation[];
}

export interface SimulationConfig {
  random_seed: number;
  ppo_max_steps: number;
  ppo_episodes: number;
  ppo_noise_low: number;
  ppo_noise_high: number;
}

export interface UncertaintyConfig {
  processing: {
    low: number;
    high: number;
  };
  transport: {
    low: number;
    high: number;
  };
  breakdown: {
    probability: number;
    repair_time: number;
  };
  vehicle_delay: {
    probability: number;
    low: number;
    high: number;
  };
}

export interface ScheduleBaseData {
  factory_info: FactoryInfo;
  shop_floor: ShopFloor;
  jobs: Job[];
  simulation_config: SimulationConfig;
  uncertainty_config: UncertaintyConfig;
  source_text: string;
  algorithm_preference?: "GA" | "PPO";
}

export type UnifiedRunMode = "compare_all" | "ppo_plan" | "ga_plan";

export type PlanEndpoint = "/run" | "/schedule/run" | "/simulation/ppo-plan";
