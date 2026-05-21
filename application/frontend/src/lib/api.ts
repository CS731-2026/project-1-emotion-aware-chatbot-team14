import { PUBLIC_BACKEND_URL } from "$env/static/public";
import type { PageSpec } from "$lib/conversation/sampleCheckIns";

const BASE = PUBLIC_BACKEND_URL || "http://localhost:3001";

export type Profile = { id: string; name: string; createdAt: string };
export type Message = { id: string; role: "user" | "agent"; content: string; timestamp: string };

// Must stay in sync with model_service/core/llm/reasoning_agent.py.
export type Mode = "qa" | "feedback" | "consent" | "done";
export type Stage = "open" | "explore" | "ground" | "close";

export type ChatDebug = {
  provider: string | null;
  model: string | null;
  current_message: string;
  mode?: Mode;
  stage?: Stage | null;
  intention?: string | null;
  system_prompt: string | null;
  history_window: number;
  history_messages: Array<{ role: string; content: string }>;
  emotional_context: string;
  transcript_lines: string[];
  prompt_messages: Array<{ role: string; content: string }>;
};

/**
 * What the frontend should render. Returned by /chat as `view`. Backend
 * (the session conductor) is the source of truth — the frontend is a
 * dumb mirror.
 */
export type ChatView = {
  surface: "chat" | "checkin" | "done";
  /** Form to mount when surface === "checkin". Backend's Pydantic mirror. */
  spec?: PageSpec | null;
  intention?: string | null;       // debug-only — never shown to the user
  state_name?: string | null;       // debug-only — internal state id
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
  sendChat: (
    text: string,
    mode?: Mode,
    stage?: Stage | null,
    opts?: { formComplete?: boolean },
  ) =>
    fetch(`${BASE}/api/v1/chat`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode, stage, form_complete: opts?.formComplete === true }),
    }).then((r) => r.json()) as Promise<{
      response: string;
      view: ChatView;
      next_mode: Mode;
      next_stage: Stage | null;
      debug: ChatDebug | null;
    }>,
};
