import express from "express";
import session from "express-session";
import { errorHandler } from "./middleware/errorHandler";
import apiRoutes from "./routes";
import env from "./config/env";

const app = express();

app.use(express.json());
app.use(
  session({
    secret: env.SESSION_SECRET,
    resave: false,
    saveUninitialized: true,
  })
);

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.use("/api/v1", apiRoutes);

app.use(errorHandler);

export default app;
