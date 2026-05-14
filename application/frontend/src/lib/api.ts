import { PUBLIC_BACKEND_URL } from "$env/static/public";

const BASE = PUBLIC_BACKEND_URL || "http://localhost:3001";

export type Profile = { id: string; name: string; createdAt: string };
export type Message = { id: string; role: "user" | "agent"; content: string; timestamp: string };
export type ChatDebug = {
  provider: string | null;
  model: string | null;
  current_message: string;
  system_prompt: string | null;
  history_window: number;
  history_messages: Array<{ role: string; content: string }>;
  emotional_context: string;
  transcript_lines: string[];
  prompt_messages: Array<{ role: string; content: string }>;
};

export const api = {
  getHealth: () => fetch(`${BASE}/health`, { credentials: "include" }).then((r) => r.json()) as Promise<{ status: string }>,
  getSession: () => fetch(`${BASE}/api/v1/session`, { credentials: "include" }).then((r) => r.json()) as Promise<{ profileId: string | null }>,
  getProfiles: () => fetch(`${BASE}/api/v1/profiles`, { credentials: "include" }).then((r) => r.json()) as Promise<Profile[]>,
  createProfile: (name: string) =>
    fetch(`${BASE}/api/v1/profiles`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).then((r) => r.json()) as Promise<Profile>,
  selectProfile: (id: string) =>
    fetch(`${BASE}/api/v1/profiles/${id}/select`, { method: "POST", credentials: "include" }).then((r) => r.json()),
  getHistory: () => fetch(`${BASE}/api/v1/history`, { credentials: "include" }).then((r) => r.json()) as Promise<Message[]>,
  sendChat: (text: string) =>
    fetch(`${BASE}/api/v1/chat`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }).then((r) => r.json()) as Promise<{ response: string; debug: ChatDebug | null }>,
};
