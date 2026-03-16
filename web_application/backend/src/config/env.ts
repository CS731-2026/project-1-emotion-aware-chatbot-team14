const env = {
  PORT: Number(process.env.PORT) || 3000,
  NODE_ENV: process.env.NODE_ENV ?? "development",
  MODEL_SERVICE_URL: process.env.MODEL_SERVICE_URL ?? "http://localhost:8000",
} as const;

export default env;
