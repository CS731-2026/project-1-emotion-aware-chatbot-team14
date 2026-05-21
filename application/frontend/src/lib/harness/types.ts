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

export const FRAME_INTERVAL_MS = 100;
export const AUDIO_CHUNK_SIZE = 512;

// ---------- VAD tuning ----------
// Voice Activity Detection runs over PCM frames captured by the AudioWorklet.
// We always capture; an utterance is recognised when a sliding window of
// recent frame-RMS values has enough above-threshold frames, and finalised
// when the ratio drops below the stop threshold for SILENCE_HOLDOFF_MS.
export const SAMPLE_RATE = 16000;
export const SPEECH_THRESHOLD = 0.006;
// Ring buffer of recent RMS values used by the windowed analysis. At 128
// samples / 16kHz ≈ 8ms per frame; 50 frames ≈ 400ms of history.
export const VAD_WINDOW_FRAMES = 50;
// Fraction of the recent window that must be above SPEECH_THRESHOLD to START
// an utterance. Higher = stricter; we want sustained speech, not a brief
// ambient spike, before we begin capture.
export const VAD_START_RATIO = 0.55;
// Fraction below which we consider the utterance trailing off.
export const VAD_STOP_RATIO = 0.1;
// Once below VAD_STOP_RATIO this long, finalise and send.
export const SILENCE_HOLDOFF_MS = 500;
// Pre-roll: how much audio before the detected utterance start to include.
export const PRE_ROLL_MS = 500;
// Minimum captured utterance duration (including pre-roll) before we'll send.
export const MIN_SPEECH_DURATION_MS = 600;
// Hard cap to avoid runaway captures.
export const MAX_SPEECH_DURATION_MS = 30000;
// Kept for backward-compat with anything still importing it; not used by the
// new VAD path.
export const SILENCE_DURATION_MS = SILENCE_HOLDOFF_MS;

export function formatTimings(timings: Record<string, number> | null | undefined) {
  if (!timings || Object.keys(timings).length === 0) return "";
  return Object.entries(timings)
    .map(([key, value]) => `${key} ${value}ms`)
    .join(", ");
}
