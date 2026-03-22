import { requestJson, getWsUrl } from "./client";
import type { PlanEndpoint, ScheduleBaseData, ScheduleMetrics, ScheduleResult, UnifiedRunMode } from "../types";

interface BackendResponse {
  message?: string;
  reply?: string;
  result?: string;
  plan?: unknown;
  schedule?: unknown;
  strategies?: unknown;
  comparison?: unknown;
  summary?: unknown;
  metrics?: {
    makespan?: number;
    vehicle_utilization?: number;
    vehicleUtilization?: number;
    transport_wait_time?: number;
    transportWaitTime?: number;
  };
  makespan?: number;
  vehicle_utilization?: number;
  transport_wait_time?: number;
  [key: string]: unknown;
}

function normalizeText(text: string): string {
  return text
    .replace(/\\r\\n|\\n|\\r/g, "\n")
    .replace(/\\t/g, " ")
    .replace(/[{}[\]]/g, " ")
    .replace(/\s+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

function flattenToLines(value: unknown, prefix = ""): string[] {
  if (value === null || value === undefined) {
    return [];
  }
  if (typeof value === "string") {
    const cleaned = normalizeText(value);
    if (!cleaned) {
      return [];
    }
    return prefix ? [`${prefix} ${cleaned}`.trim()] : [cleaned];
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return [prefix ? `${prefix} ${String(value)}`.trim() : String(value)];
  }
  if (Array.isArray(value)) {
    if (!value.length) {
      return [];
    }
    return value.flatMap((item, index) => flattenToLines(item, prefix ? `${prefix} #${index + 1}` : `#${index + 1}`));
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (!entries.length) {
      return [];
    }
    return entries.flatMap(([key, child]) => flattenToLines(child, prefix ? `${prefix} ${key}:` : `${key}:`));
  }
  return [prefix ? `${prefix} ${String(value)}`.trim() : String(value)];
}

function toReadableText(value: unknown): string {
  const lines = flattenToLines(value);
  if (!lines.length) {
    return "后端返回了空内容。";
  }
  const text = lines.join("\n");
  return text.length > 8000 ? `${text.slice(0, 8000)}\n...（内容过长，已截断）` : text;
}

function resolveReplyText(data: BackendResponse): string {
  const directText = [data.message, data.reply, data.result].find((item) => typeof item === "string" && item.trim());
  if (directText) {
    const raw = directText.trim();
    if (raw.startsWith("{") || raw.startsWith("[")) {
      try {
        return toReadableText(JSON.parse(raw));
      } catch {
        return normalizeText(directText);
      }
    }
    return normalizeText(directText);
  }
  const content = data.summary ?? data.comparison ?? data.strategies ?? data.plan ?? data.schedule ?? data;
  if (typeof content === "string") {
    return normalizeText(content);
  }
  return toReadableText(content);
}

function pickNumberByPath(data: unknown, paths: string[][]): number | undefined {
  for (const path of paths) {
    let current: unknown = data;
    for (const key of path) {
      if (!current || typeof current !== "object" || !(key in current)) {
        current = undefined;
        break;
      }
      current = (current as Record<string, unknown>)[key];
    }
    const value = Number(current);
    if (Number.isFinite(value)) {
      return value;
    }
  }
  return undefined;
}

function normalizeMetrics(data: BackendResponse): ScheduleMetrics | undefined {
  const makespan = pickNumberByPath(data, [
    ["metrics", "makespan"],
    ["summary", "metrics", "makespan"],
    ["comparison", "best_metrics", "makespan"],
    ["comparison", "metrics", "makespan"],
    ["makespan"]
  ]);
  const vehicleUtilization = pickNumberByPath(data, [
    ["metrics", "vehicle_utilization"],
    ["metrics", "vehicleUtilization"],
    ["summary", "metrics", "vehicle_utilization"],
    ["summary", "metrics", "vehicleUtilization"],
    ["comparison", "best_metrics", "vehicle_utilization"],
    ["comparison", "best_metrics", "vehicleUtilization"],
    ["vehicle_utilization"],
    ["vehicleUtilization"]
  ]);
  const transportWaitTime = pickNumberByPath(data, [
    ["metrics", "transport_wait_time"],
    ["metrics", "transportWaitTime"],
    ["summary", "metrics", "transport_wait_time"],
    ["summary", "metrics", "transportWaitTime"],
    ["comparison", "best_metrics", "transport_wait_time"],
    ["comparison", "best_metrics", "transportWaitTime"],
    ["transport_wait_time"],
    ["transportWaitTime"]
  ]);

  const result: ScheduleMetrics = {};
  if (Number.isFinite(makespan)) {
    result.makespan = makespan;
  }
  if (Number.isFinite(vehicleUtilization)) {
    result.vehicleUtilization = vehicleUtilization;
  }
  if (Number.isFinite(transportWaitTime)) {
    result.transportWaitTime = transportWaitTime;
  }
  if (!Object.keys(result).length) {
    return undefined;
  }
  return result;
}

interface SendTextOptions {
  mode?: UnifiedRunMode;
  ppoPolicyId?: string;
}

function buildUnifiedPayload(prompt: string, baseData?: ScheduleBaseData, options?: SendTextOptions) {
  return {
    mode: options?.mode ?? "compare_all",
    factory_info: baseData?.factory_info,
    shop_floor: baseData?.shop_floor,
    jobs: baseData?.jobs,
    simulation_config: baseData?.simulation_config,
    uncertainty_config: baseData?.uncertainty_config,
    dispatching_config: {
      ppo_policy_id: options?.ppoPolicyId?.trim() || "latest"
    },
    source_text: baseData?.source_text ?? prompt,
    algorithm_preference: baseData?.algorithm_preference,
    prompt,
    message: prompt,
    query: prompt
  };
}

export async function sendTextToSchedule(
  endpoint: PlanEndpoint,
  prompt: string,
  baseData?: ScheduleBaseData,
  options?: SendTextOptions
): Promise<ScheduleResult> {
  const payload =
    endpoint === "/run"
      ? buildUnifiedPayload(prompt, baseData, options)
      : {
          prompt,
          message: prompt,
          query: prompt,
          baseData,
          basicData: baseData,
          extracted: baseData
        };
  const data = await requestJson<BackendResponse>(endpoint, {
    method: "POST",
    body: JSON.stringify(payload),
    headers: {
      "Content-Type": "application/json"
    }
  });

  return {
    replyText: resolveReplyText(data),
    plan: data.plan ?? data.schedule ?? data.strategies ?? data.comparison ?? data.summary ?? data,
    metrics: normalizeMetrics(data),
    raw: data
  };
}

export async function uploadImage(file: File): Promise<ScheduleResult> {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("file", file);

  const data = await requestJson<BackendResponse>("/upload_image", {
    method: "POST",
    body: formData
  });

  return {
    replyText: resolveReplyText(data),
    plan: data.plan ?? data.schedule,
    metrics: normalizeMetrics(data),
    raw: data
  };
}

export function createProgressSocket(onMessage: (value: string) => void): WebSocket {
  const socket = new WebSocket(getWsUrl("/ws/progress"));
  socket.onmessage = (event) => {
    onMessage(String(event.data ?? ""));
  };
  return socket;
}
