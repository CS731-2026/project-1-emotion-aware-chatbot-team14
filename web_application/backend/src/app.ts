import express from "express";
import { errorHandler } from "./middleware/errorHandler";
import apiRoutes from "./routes";

const app = express();

app.use(express.json());

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.use("/api/v1", apiRoutes);

app.use(errorHandler);

export default app;
