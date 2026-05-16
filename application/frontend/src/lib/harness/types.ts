// EmpathBot 6-class schema — must mirror application/model_service/core/emotion/base.py.
export type Emotion =
  | "neutral"
  | "trust_relief"
  | "sadness"
  | "fear_anxiety"
  | "confusion"
  | "distrust";

export type TranscriptEntry = {
  id: string;
  text: string;
  timestamp: string;
  chunk: number | null;
  error: string | null;
  timings: string;
};

export const EMOTION_COLOURS: Record<Emotion, string> = {
  neutral:      "hsl(220, 15%, 10%)",
  trust_relief: "hsl(150, 45%, 14%)",
  sadness:      "hsl(210, 60%, 12%)",
  fear_anxiety: "hsl(275, 50%, 13%)",
  confusion:    "hsl(45, 55%, 14%)",
  distrust:     "hsl(0, 55%, 13%)",
};

export const EMOTION_LABEL: Record<Emotion, string> = {
  neutral:      "Neutral",
  trust_relief: "Calm / Reassured",
  sadness:      "Sad",
  fear_anxiety: "Anxious",
  confusion:    "Confused",
  distrust:     "Wary",
};

export function isEmotion(value: unknown): value is Emotion {
  return typeof value === "string" && value in EMOTION_COLOURS;
}

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
