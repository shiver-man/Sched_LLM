import { requestJson } from "./client";
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
  experience_id?: string;
  reflection?: string;
  llm_readable_brief?: string;
  summary_comparison?: unknown;
  detailed_schemes?: unknown;
  gantt_chart_base64?: string;
  status?: string;
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
  return lines.join("\n");
}

function resolveReplyText(data: BackendResponse): string {
  if (data.llm_readable_brief) {
    return normalizeText(data.llm_readable_brief);
  }
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

function pickBestSummaryRow(summary: unknown): Record<string, unknown> | undefined {
  if (!Array.isArray(summary)) {
    return undefined;
  }

  const rows = summary.filter((item): item is Record<string, unknown> => !!item && typeof item === "object");
  if (!rows.length) {
    return undefined;
  }

  const completeRow = rows.find((item) => item.is_complete === true);
  return completeRow ?? rows[0];
}

function normalizeMetrics(data: BackendResponse): ScheduleMetrics | undefined {
  const bestSummaryRow = pickBestSummaryRow(data.summary_comparison);
  const makespan = pickNumberByPath(data, [
    ["metrics", "makespan"],
    ["summary", "metrics", "makespan"],
    ["comparison", "best_metrics", "makespan"],
    ["comparison", "metrics", "makespan"],
    ["best_strategy_result", "metrics", "makespan"],
    ["best_result", "metrics", "makespan"],
    ["best_metrics", "makespan"],
    ["makespan"]
  ]) ?? pickNumberByPath(bestSummaryRow, [["makespan"]]);
  const vehicleUtilization = pickNumberByPath(data, [
    ["metrics", "vehicle_utilization"],
    ["metrics", "vehicleUtilization"],
    ["summary", "metrics", "vehicle_utilization"],
    ["summary", "metrics", "vehicleUtilization"],
    ["comparison", "best_metrics", "vehicle_utilization"],
    ["comparison", "best_metrics", "vehicleUtilization"],
    ["best_strategy_result", "metrics", "vehicle_utilization"],
    ["best_strategy_result", "metrics", "vehicleUtilization"],
    ["best_result", "metrics", "vehicle_utilization"],
    ["best_metrics", "vehicle_utilization"],
    ["vehicle_utilization"],
    ["vehicleUtilization"]
  ]) ?? pickNumberByPath(bestSummaryRow, [
    ["vehicle_utilization"],
    ["vehicleUtilization"],
    ["utilization"]
  ]);
  const transportWaitTime = pickNumberByPath(data, [
    ["metrics", "transport_wait_time"],
    ["metrics", "transportWaitTime"],
    ["summary", "metrics", "transport_wait_time"],
    ["summary", "metrics", "transportWaitTime"],
    ["comparison", "best_metrics", "transport_wait_time"],
    ["comparison", "best_metrics", "transportWaitTime"],
    ["best_strategy_result", "metrics", "transport_wait_time"],
    ["best_strategy_result", "metrics", "transportWaitTime"],
    ["best_result", "metrics", "transport_wait_time"],
    ["best_metrics", "transport_wait_time"],
    ["transport_wait_time"],
    ["transportWaitTime"]
  ]) ?? pickNumberByPath(bestSummaryRow, [["transport_wait_time"], ["transportWaitTime"]]);

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
  useLlm?: boolean;
}

function buildUnifiedPayload(prompt: string, baseData?: ScheduleBaseData, options?: SendTextOptions) {
  const payload = {
    mode: options?.mode ?? "compare_all",
    factory_info: baseData?.factory_info,
    shop_floor: baseData?.shop_floor,
    jobs: baseData?.jobs || [],
    simulation_config: baseData?.simulation_config,
    uncertainty_config: baseData?.uncertainty_config,
    dispatching_config: {
      ppo_policy_id: options?.ppoPolicyId?.trim() || baseData?.dispatching_config?.ppo_policy_id || "latest"
    },
    llm_config: {
      use_ollama: options?.useLlm ?? baseData?.llm_config?.use_ollama ?? true
    },
    return_raw_json: baseData?.return_raw_json ?? true,
    source_text: baseData?.source_text ?? prompt,
    algorithm_preference: baseData?.algorithm_preference,
    prompt,
    message: prompt,
    query: prompt
  };

  console.log("Sending payload to backend:", payload);
  return payload;
}

export async function sendTextToSchedule(
  _endpoint: PlanEndpoint,
  prompt: string,
  baseData?: ScheduleBaseData,
  options?: SendTextOptions
): Promise<ScheduleResult> {
  const payload = buildUnifiedPayload(prompt, baseData, options);

  const data = await requestJson<BackendResponse>("/run", {
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
    experience_id: data.experience_id,
    reflection: data.reflection,
    llm_readable_brief: data.llm_readable_brief,
    summary_comparison: data.summary_comparison,
    detailed_schemes: data.detailed_schemes,
    gantt_chart_base64: data.gantt_chart_base64,
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
