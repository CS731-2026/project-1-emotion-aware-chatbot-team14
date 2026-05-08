import "dotenv/config";
import app from "./app";
import env from "./config/env";

const server = app.listen(env.PORT, () => {
  console.log(`Backend running on port ${env.PORT} [${env.NODE_ENV}]`);
});

server.on("error", (err: NodeJS.ErrnoException) => {
  if (err.code === "EADDRINUSE") {
    console.error(
      `Backend could not start because port ${env.PORT} is already in use. ` +
      `Run "make kill" or stop the existing process on that port, then try "make dev" again.`
    );
    process.exit(1);
  }

  throw err;
});
