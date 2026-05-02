export type Emotion =
  | "neutral"
  | "happy"
  | "sad"
  | "angry"
  | "fearful"
  | "disgusted"
  | "surprised";

export type TranscriptEntry = {
  id: string;
  text: string;
  timestamp: string;
  chunk: number | null;
  error: string | null;
  timings: string;
};

export const EMOTION_COLOURS: Record<Emotion, string> = {
  neutral: "hsl(220, 15%, 10%)",
  happy: "hsl(45, 70%, 15%)",
  sad: "hsl(210, 60%, 12%)",
  angry: "hsl(0, 65%, 15%)",
  fearful: "hsl(275, 50%, 13%)",
  disgusted: "hsl(120, 35%, 11%)",
  surprised: "hsl(180, 55%, 13%)",
};

export const EMOTION_MAP: Record<string, Emotion> = {
  angry: "angry",
  disgust: "disgusted",
  fear: "fearful",
  happy: "happy",
  sad: "sad",
  surprise: "surprised",
  neutral: "neutral",
};

export const FRAME_INTERVAL_MS = 500;
export const AUDIO_CHUNK_SIZE = 512;
export const SPEECH_THRESHOLD = 0.00075;
export const SILENCE_DURATION_MS = 600;
export const MIN_SPEECH_DURATION_MS = 1200;
export const MAX_SPEECH_DURATION_MS = 30000;

export function formatTimings(timings: Record<string, number> | null | undefined) {
  if (!timings || Object.keys(timings).length === 0) return "";
  return Object.entries(timings)
    .map(([key, value]) => `${key} ${value}ms`)
    .join(", ");
}
