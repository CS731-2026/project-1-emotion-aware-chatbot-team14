import { Router } from "express";
import predictionRouter from "./prediction.router";

const router = Router();

router.use("/predict", predictionRouter);

export default router;
