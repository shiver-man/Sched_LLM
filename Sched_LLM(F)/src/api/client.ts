const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim() || "http://127.0.0.1:8000";

type HttpMethod = "GET" | "POST";

interface RequestOptions {
  method?: HttpMethod;
  body?: BodyInit | null;
  headers?: Record<string, string>;
}

export async function requestJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: options.method ?? "GET",
    body: options.body,
    headers: {
      ...(options.headers ?? {})
    }
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`请求失败 ${response.status}: ${errorText}`);
  }

  return (await response.json()) as T;
}

export function getWsUrl(path: string): string {
  const url = new URL(API_BASE_URL);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = path;
  url.search = "";
  return url.toString();
}
