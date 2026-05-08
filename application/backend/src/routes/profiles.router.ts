import { Router } from "express";
import { listProfiles, createProfile, getProfile } from "../lib/profileStore";
import type { AppError } from "../middleware/errorHandler";

const router = Router();

router.get("/", (_req, res) => {
  res.json(listProfiles());
});

router.post("/", (req, res, next) => {
  try {
    const { name } = req.body as { name?: string };
    if (!name || typeof name !== "string" || name.trim() === "") {
      const err: AppError = Object.assign(new Error("name is required"), { statusCode: 400 });
      return next(err);
    }
    const profile = createProfile(name.trim());
    res.status(201).json(profile);
  } catch (err) {
    next(err);
  }
});

router.post("/:id/select", (req, res, next) => {
  try {
    const profile = getProfile(req.params.id);
    if (!profile) {
      const err: AppError = Object.assign(new Error("Profile not found"), { statusCode: 404 });
      return next(err);
    }
    req.session.profileId = profile.id;
    res.json({ ok: true });
  } catch (err) {
    next(err);
  }
});

export default router;
