const env = {
  PORT: Number(process.env.PORT) || 3000,
  NODE_ENV: process.env.NODE_ENV ?? "development",
  MODEL_SERVICE_URL: process.env.MODEL_SERVICE_URL ?? "http://localhost:8000",
  FRONTEND_ORIGIN: process.env.FRONTEND_ORIGIN ?? "http://localhost:5173",
  SESSION_SECRET: process.env.SESSION_SECRET ?? "hri-dev-secret-change-in-prod",
} as const;

export default env;
