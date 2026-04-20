import { BACKEND_URL } from "$env/static/private";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async () => {
  const response = await fetch(`${BACKEND_URL}/api/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input: "test" }),
  });

  const prediction = await response.json();

  return {
    message: "Hello from the SvelteKit server",
    timestamp: new Date().toISOString(),
    prediction,
  };
};
