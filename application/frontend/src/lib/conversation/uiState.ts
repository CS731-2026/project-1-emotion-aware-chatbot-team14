import type { TranscriptEntry } from "$lib/harness/types";

export type AssistantPhase =
  | "offline"
  | "needs-profile"
  | "thinking"
  | "speaking"
  | "listening"
  | "ready";

export type AssistantPhaseInput = {
  backendOnline: boolean;
  profileReady: boolean;
  isListening: boolean;
  chatBusy: boolean;
  isSpeaking: boolean;
};

export function deriveAssistantPhase(input: AssistantPhaseInput): AssistantPhase {
  if (!input.backendOnline) return "offline";
  if (!input.profileReady) return "needs-profile";
  if (input.chatBusy) return "thinking";
  if (input.isSpeaking) return "speaking";
  if (input.isListening) return "listening";
  return "ready";
}

export function phaseLabel(phase: AssistantPhase): string {
  switch (phase) {
    case "offline":
      return "Backend offline";
    case "needs-profile":
      return "Choose a profile";
    case "thinking":
      return "Forming a response";
    case "speaking":
      return "Speaking";
    case "listening":
      return "Listening";
    case "ready":
      return "Ready";
  }
}

function isPlaceholderTranscript(text: string): boolean {
  const normalized = text.trim().toLowerCase();
  return normalized === ""
    || normalized === "listening for harness transcript..."
    || normalized === "mic idle"
    || normalized.startsWith("sent speech clip ")
    || normalized === "harness received audio but produced no transcript";
}

export function transcriptPreview(
  liveTranscript: string,
  entries: TranscriptEntry[],
): string {
  if (!isPlaceholderTranscript(liveTranscript)) {
    return liveTranscript.trim();
  }

  const latestRealEntry = entries.find((entry) => entry.text.trim() !== "" && entry.text !== "[no transcript text]");
  if (latestRealEntry) return latestRealEntry.text.trim();

  return "Start speaking when you're ready.";
}

export function shouldPromoteTranscript(text: string, previousTranscript: string | null): boolean {
  const normalized = text.trim();
  if (normalized === "" || normalized === previousTranscript) return false;
  return !isPlaceholderTranscript(normalized);
}
