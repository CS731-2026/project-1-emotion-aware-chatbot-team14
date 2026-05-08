import { Router } from "express";
import type { Request, Response, NextFunction } from "express";
import env from "../config/env";

const router = Router();

/**
 * POST /api/v1/predict
 * Forwards a prediction request to the model service.
 * Body: { input: unknown }
 */
router.post(
  "/",
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      try {
        const response = await fetch(`${env.MODEL_SERVICE_URL}/api/v1/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(req.body),
        });

        const data = await response.json();
        return res.status(response.status).json(data);
      } catch {
        return res.json({
          message: "predict fallback from backend",
          backend: "ok",
          modelService: "offline",
          input: req.body?.input ?? null,
        });
      }
    } catch (err) {
      next(err);
    }
  }
);

export default router;
