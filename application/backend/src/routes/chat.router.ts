import { Router } from "express";
import { getHistory, appendMessage } from "../lib/profileStore";
import type { AppError } from "../middleware/errorHandler";
import type { Message } from "../types/profile";
import env from "../config/env";
import { randomUUID } from "crypto";

const router = Router();

const HISTORY_WINDOW = 10;

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

    const harnessRes = await fetch(`${env.MODEL_SERVICE_URL}/api/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile_id: profileId, message: text, history: harnessHistory }),
    });

    if (!harnessRes.ok) {
      const err: AppError = Object.assign(new Error("Harness error"), { statusCode: 502 });
      return next(err);
    }

    const { response } = (await harnessRes.json()) as { response: string };

    const userMsg: Message = { id: randomUUID(), role: "user", content: text, timestamp: new Date().toISOString() };
    const agentMsg: Message = { id: randomUUID(), role: "agent", content: response, timestamp: new Date().toISOString() };

    appendMessage(profileId, userMsg);
    appendMessage(profileId, agentMsg);

    res.json({ response });
  } catch (err) {
    next(err);
  }
});

export default router;
