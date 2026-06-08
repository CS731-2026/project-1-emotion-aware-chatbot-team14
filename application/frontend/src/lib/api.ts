import { PUBLIC_BACKEND_URL } from "$env/static/public";
import type { PageSpec } from "$lib/conversation/sampleCheckIns";

const BASE = PUBLIC_BACKEND_URL || "http://localhost:3001";

export type Profile = { id: string; name: string; createdAt: string };
export type Message = { id: string; role: "user" | "agent"; content: string; timestamp: string };

export type ChatDebug = {
  provider: string | null;
  model: string | null;
  current_message: string;
  intention?: string | null;
  system_prompt: string | null;
  history_window: number;
  history_messages: Array<{ role: string; content: string }>;
  emotional_context: string;
  transcript_lines: string[];
  prompt_messages: Array<{ role: string; content: string }>;
  /** Conductor view, current state + facts. Populated when a session is active. */
  session_state?: {
    state_name: string | null;
    surface: "chat" | "checkin" | "done";
    turn_in_state: number;
    segment_id: number;
    emissions: Array<{ name: string; payload: Record<string, unknown> }>;
    state_facts: Record<string, Record<string, unknown>>;
  };
};

/**
 * What the frontend should render. Returned by /chat as `view`. Backend
 * (the session conductor) is the source of truth, the frontend is a
 * dumb mirror.
 */
export type ChatView = {
  surface: "chat" | "checkin" | "done";
  /** Form to mount when surface === "checkin". Backend's Pydantic mirror. */
  spec?: PageSpec | null;
  intention?: string | null;       // debug-only, never shown to the user
  state_name?: string | null;       // debug-only, internal state id
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
  appendHistory: (message: Message) =>
    fetch(`${BASE}/api/v1/history`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message),
    }).then((r) => r.json()) as Promise<{ ok: boolean }>,
  sendChat: (text: string) =>
    fetch(`${BASE}/api/v1/chat`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }).then((r) => r.json()) as Promise<{
      response: string;
      view: ChatView;
      debug: ChatDebug | null;
    }>,
};
