import { Router } from "express";
import { getHistory, appendMessage } from "../lib/profileStore";
import type { AppError } from "../middleware/errorHandler";
import type { Message } from "../types/profile";
import env from "../config/env";
import { randomUUID } from "crypto";

const router = Router();

// Number of prior turns (user + agent pairs) sent to the model service.
const HISTORY_WINDOW = 10;

// Must stay in sync with model_service/core/llm/reasoning_agent.py.
type Mode = "qa" | "feedback" | "consent" | "done";
type Stage = "open" | "explore" | "ground" | "close";

const VALID_MODES: ReadonlySet<Mode> = new Set(["qa", "feedback", "consent", "done"]);
const VALID_STAGES: ReadonlySet<Stage> = new Set(["open", "explore", "ground", "close"]);

/** Returns a safe fallback string when the model service is unreachable. */
function buildFallbackResponse(text: string): string {
  return `Backend fallback reply: I received "${text}". The backend is working, but the model service or LLM is unavailable.`;
}

type ChatDebug = {
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

type ChatView = {
  surface: "chat" | "checkin" | "done";
  intention?: string | null;
  state_name?: string | null;
};

// Default view used when the model service is unreachable — keep the
// frontend on the chat surface so the user can still see the fallback reply.
const FALLBACK_VIEW: ChatView = { surface: "chat" };

/**
 * POST /api/v1/chat
 * Forwards the user's message + windowed history to the model service, then
 * persists both the user turn and the agent reply to the profile store.
 * Falls back gracefully if the model service is down.
 */
router.post("/", async (req, res, next) => {
  try {
    const profileId = req.session.profileId;
    if (!profileId) {
      const err: AppError = Object.assign(new Error("No active profile"), { statusCode: 401 });
      return next(err);
    }

    const { text, mode: rawMode, stage: rawStage } = req.body as {
      text?: string;
      mode?: string;
      stage?: string | null;
    };
    if (!text || typeof text !== "string") {
      const err: AppError = Object.assign(new Error("text is required"), { statusCode: 400 });
      return next(err);
    }

    // Drop unrecognised values silently so the model service falls back to its own defaults.
    const mode: Mode = rawMode && VALID_MODES.has(rawMode as Mode) ? (rawMode as Mode) : "qa";
    const stage: Stage | null =
      rawStage && VALID_STAGES.has(rawStage as Stage) ? (rawStage as Stage) : null;

    const fullHistory = getHistory(profileId);
    const windowed = fullHistory.slice(-(HISTORY_WINDOW * 2));
    const harnessHistory = windowed.map((m) => ({ role: m.role === "agent" ? "assistant" : m.role, content: m.content }));

    let response = buildFallbackResponse(text);
    let debug: ChatDebug | null = null;
    let view: ChatView = FALLBACK_VIEW;
    // If the model service is unreachable we keep the caller's state — no transition.
    let nextMode: Mode = mode;
    let nextStage: Stage | null = stage;

    try {
      const harnessRes = await fetch(`${env.MODEL_SERVICE_URL}/api/v1/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile_id: profileId,
          message: text,
          history: harnessHistory,
          mode,
          stage,
        }),
      });

      if (harnessRes.ok) {
        const data = (await harnessRes.json()) as {
          response?: string;
          view?: ChatView;
          next_mode?: string;
          next_stage?: string | null;
          debug?: ChatDebug;
        };
        if (data.response) response = data.response;
        if (data.debug) debug = data.debug;
        if (data.view) view = data.view;
        if (data.next_mode && VALID_MODES.has(data.next_mode as Mode)) {
          nextMode = data.next_mode as Mode;
        }
        if (data.next_stage === null) {
          nextStage = null;
        } else if (data.next_stage && VALID_STAGES.has(data.next_stage as Stage)) {
          nextStage = data.next_stage as Stage;
        }
      }
    } catch {
      // Keep the backend usable locally even when the model service is down.
    }

    const userMsg: Message = { id: randomUUID(), role: "user", content: text, timestamp: new Date().toISOString() };
    const agentMsg: Message = { id: randomUUID(), role: "agent", content: response, timestamp: new Date().toISOString() };

    appendMessage(profileId, userMsg);
    appendMessage(profileId, agentMsg);

    res.json({ response, view, next_mode: nextMode, next_stage: nextStage, debug });
  } catch (err) {
    next(err);
  }
});

export default router;
