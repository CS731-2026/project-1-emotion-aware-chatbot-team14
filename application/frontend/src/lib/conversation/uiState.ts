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

export function isRealTranscript(text: string): boolean {
  return !isPlaceholderTranscript(text);
}

function isPlaceholderTranscript(text: string): boolean {
  const normalized = text.trim().toLowerCase();
  if (normalized === "") return true;
  if (normalized === "listening for harness transcript...") return true;
  if (normalized === "start speaking when you're ready.") return true;
  if (normalized === "mic idle") return true;
  if (normalized.startsWith("sent speech clip ")) return true;
  if (normalized === "harness received audio but produced no transcript") return true;
  if (normalized === "[whisper.cpp returned no text]") return true;
  if (normalized === "[whisper.cpp transcript filtered]") return true;
  if (normalized === "[no transcript text]") return true;
  // Any single bracketed token whisper emits as a non-speech tag:
  // [BLANK_AUDIO], [Music], [ Pause ], (sighs), (upbeat music), etc.
  if (/^\[[a-z0-9_\- ]+\]$/.test(normalized)) return true;
  if (/^\([a-z0-9_\- ,]+\)$/.test(normalized)) return true;
  return false;
}

export function transcriptPreview(
  liveTranscript: string,
  entries: TranscriptEntry[],
): string {
  if (isRealTranscript(liveTranscript)) {
    return liveTranscript.trim();
  }
  // Fallback: walk the recent entries for a real transcript. Skip every kind
  // of placeholder, [whisper.cpp transcript filtered], [BLANK_AUDIO],
  // [no transcript text], (sighs), etc., not just the empty/no-text ones.
  const latestRealEntry = entries.find((entry) => isRealTranscript(entry.text));
  if (latestRealEntry) return latestRealEntry.text.trim();

  return "Start speaking when you're ready.";
}

export function shouldPromoteTranscript(text: string, previousTranscript: string | null): boolean {
  const normalized = text.trim();
  if (normalized === "" || normalized === previousTranscript) return false;
  return !isPlaceholderTranscript(normalized);
}
