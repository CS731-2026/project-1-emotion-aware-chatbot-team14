import { Router } from "express";
import { getHistory, appendMessage } from "../lib/profileStore";
import type { AppError } from "../middleware/errorHandler";
import type { Message } from "../types/profile";

const router = Router();

router.get("/", (req, res, next) => {
  try {
    const profileId = req.session.profileId;
    if (!profileId) {
      const err: AppError = Object.assign(new Error("No active profile"), { statusCode: 401 });
      return next(err);
    }
    res.json(getHistory(profileId));
  } catch (err) {
    next(err);
  }
});

router.post("/", (req, res, next) => {
  try {
    const profileId = req.session.profileId;
    if (!profileId) {
      const err: AppError = Object.assign(new Error("No active profile"), { statusCode: 401 });
      return next(err);
    }
    const message = req.body as Message;
    appendMessage(profileId, message);
    res.json({ ok: true });
  } catch (err) {
    next(err);
  }
});

export default router;
