import type { ChatMessage, ScheduleMetrics } from "../types";

const SAMPLE_IMAGE_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAQAAAAAYLlVAAAA8ElEQVR4Ae3XwQ3CMBQF0M8nGvT9d2YQ2WwVgS5NQ22EJf0Rt6R2wqQkV9+wsH5nH0wA3bQ4R3i6r3gM8Q6O6k+fKpYw0mJ8L7xqg4x+3cQm2mH2kM3r6k0f0wC2wHf9QJ4gT8j3YQH5+gBv8h6W3s0fQn7Qw3YV+2XW8m4R4m+XnJ8X9Sx3J5W4i3n+Jm0z3l9Xf0QH+Q+fB1V7Wv2mE4jz6w8L8c9K6S7x8J7g3p2xS6b1m8o9G+v7E2k7r4A5k8I4k7X5F0RjS+4zqC6kW7uW5C3UOQ3hA8QmM7U8m8O5c0P9i8g9QJXj0r1j4Q9cAAAAASUVORK5CYII=";

export const initialMessages: ChatMessage[] = [
  {
    id: "m-1",
    role: "assistant",
    contentType: "text",
    content:
      "Hi, I am your Sched_LLM scheduling assistant. To get better planning results, please provide these fields in plain language: factory_info (factory_id, factory_name, planning_horizon, current_time), shop_floor.machines (machine_id, location, status, type), shop_floor.vehicles (vehicle_id, start_location, speed, load_unload_time, status), transport_network (nodes, edges with from/to/distance), and jobs (job_id, release_time, due_time, initial_location, operations with candidate_machines). It is strongly recommended to also include simulation_config (random_seed, ppo_max_steps, ppo_episodes, ppo_noise_low/high) and uncertainty_config (processing, transport, breakdown, vehicle_delay).",
    createdAt: Date.now() - 120000
  },
  {
    id: "m-2",
    role: "user",
    contentType: "text",
    content: "车间 A 有 12 台设备，3 辆 AGV，请生成 PPO 调度计划。",
    createdAt: Date.now() - 90000
  },
  {
    id: "m-3",
    role: "assistant",
    contentType: "text",
    content: "已收到。你可以继续补充约束，例如交付时间窗、换线时间和任务优先级。",
    createdAt: Date.now() - 60000
  },
  {
    id: "m-4",
    role: "user",
    contentType: "image",
    content: SAMPLE_IMAGE_DATA_URL,
    createdAt: Date.now() - 30000
  }
];

export const initialMetrics: ScheduleMetrics = {
  makespan: 428,
  vehicleUtilization: 0.81,
  transportWaitTime: 17.3
};
