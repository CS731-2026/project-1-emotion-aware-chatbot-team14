import { Router } from "express";
import { getHistory, appendMessage } from "../lib/profileStore";
import type { AppError } from "../middleware/errorHandler";
import type { Message } from "../types/profile";
import env from "../config/env";
import { randomUUID } from "crypto";

const router = Router();

// Number of prior turns (user + agent pairs) sent to the model service.
const HISTORY_WINDOW = 10;

/** Returns a safe fallback string when the model service is unreachable. */
function buildFallbackResponse(text: string): string {
  return `Backend fallback reply: I received "${text}". The backend is working, but the model service or LLM is unavailable.`;
}

type ChatDebug = {
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

    const { text } = req.body as { text?: string };
    if (!text || typeof text !== "string") {
      const err: AppError = Object.assign(new Error("text is required"), { statusCode: 400 });
      return next(err);
    }

    const fullHistory = getHistory(profileId);
    const windowed = fullHistory.slice(-(HISTORY_WINDOW * 2));
    const harnessHistory = windowed.map((m) => ({ role: m.role === "agent" ? "assistant" : m.role, content: m.content }));

    let response = buildFallbackResponse(text);
    let debug: ChatDebug | null = null;

    try {
      const harnessRes = await fetch(`${env.MODEL_SERVICE_URL}/api/v1/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile_id: profileId, message: text, history: harnessHistory }),
      });

      if (harnessRes.ok) {
        const data = (await harnessRes.json()) as { response?: string; debug?: ChatDebug };
        if (data.response) response = data.response;
        if (data.debug) debug = data.debug;
      }
    } catch {
      // Keep the backend usable locally even when the model service is down.
    }

    const userMsg: Message = { id: randomUUID(), role: "user", content: text, timestamp: new Date().toISOString() };
    const agentMsg: Message = { id: randomUUID(), role: "agent", content: response, timestamp: new Date().toISOString() };

    appendMessage(profileId, userMsg);
    appendMessage(profileId, agentMsg);

    res.json({ response, debug });
  } catch (err) {
    next(err);
  }
});

export default router;
