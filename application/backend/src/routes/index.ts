import { Router } from "express";
import predictionRouter from "./prediction.router";
import profilesRouter from "./profiles.router";
import historyRouter from "./history.router";
import sessionRouter from "./session.router";
import chatRouter from "./chat.router";

const router = Router();

router.use("/predict", predictionRouter);
router.use("/profiles", profilesRouter);
router.use("/history", historyRouter);
router.use("/session", sessionRouter);
router.use("/chat", chatRouter);

export default router;
