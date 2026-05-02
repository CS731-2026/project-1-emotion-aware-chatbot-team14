import express from "express";
import session from "express-session";
import type { Request, Response, NextFunction } from "express";
import { errorHandler } from "./middleware/errorHandler";
import apiRoutes from "./routes";
import env from "./config/env";

const app = express();
const FRONTEND_ORIGIN = "http://localhost:5173";

app.use((req: Request, res: Response, next: NextFunction) => {
  res.header("Access-Control-Allow-Origin", FRONTEND_ORIGIN);
  res.header("Access-Control-Allow-Credentials", "true");
  res.header("Access-Control-Allow-Headers", "Content-Type");
  res.header("Access-Control-Allow-Methods", "GET,POST,OPTIONS");

  if (req.method === "OPTIONS") {
    return res.sendStatus(204);
  }

  next();
});

app.use(express.json());
app.use(
  session({
    secret: env.SESSION_SECRET,
    resave: false,
    saveUninitialized: true,
  })
);

app.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    service: "backend",
    modelServiceUrl: env.MODEL_SERVICE_URL,
  });
});

app.use("/api/v1", apiRoutes);

app.use(errorHandler);

export default app;
