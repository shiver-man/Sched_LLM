const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim() || "http://127.0.0.1:8000";

type HttpMethod = "GET" | "POST";

interface RequestOptions {
  method?: HttpMethod;
  body?: BodyInit | null;
  headers?: Record<string, string>;
  timeout?: number;
}

export async function requestJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const timeout = options.timeout ?? 600000; // 提升到 10 分钟 (600s)
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  // 规范化 URL，防止双斜杠
  const baseUrl = API_BASE_URL.endsWith("/") ? API_BASE_URL.slice(0, -1) : API_BASE_URL;
  const urlPath = path.startsWith("/") ? path : `/${path}`;
  const fullUrl = `${baseUrl}${urlPath}`;

  try {
    const fetchOptions: RequestInit = {
      method: options.method ?? "GET",
      body: options.body,
      signal: controller.signal
    };

    // 只有在提供了 headers 时才设置，避免干扰 FormData 的自动边界设置
    if (options.headers) {
      fetchOptions.headers = options.headers;
    }

    const response = await fetch(fullUrl, fetchOptions);

    clearTimeout(id);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`请求失败 ${response.status}: ${errorText}`);
    }

    return (await response.json()) as T;
  } catch (error) {
    clearTimeout(id);
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`请求超时：大模型分析耗时较长，请检查网络或后端状态 (timeout: ${timeout / 1000}s)`);
    }
    throw error;
  }
}

export function getWsUrl(path: string): string {
  const url = new URL(API_BASE_URL);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = path;
  url.search = "";
  return url.toString();
}
