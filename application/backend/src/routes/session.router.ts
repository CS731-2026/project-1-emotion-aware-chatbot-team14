import { Router } from "express";

const router = Router();

router.get("/", (req, res) => {
  res.json({ profileId: req.session.profileId ?? null });
});

export default router;
