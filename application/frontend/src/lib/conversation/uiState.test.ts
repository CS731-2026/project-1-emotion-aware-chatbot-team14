import { describe, expect, it } from "vitest";

import type { TranscriptEntry } from "$lib/harness/types";
import { deriveAssistantPhase, phaseLabel, transcriptPreview } from "$lib/conversation/uiState";

function entry(text: string): TranscriptEntry {
  return {
    id: crypto.randomUUID(),
    text,
    timestamp: "10:30",
    chunk: 1,
    error: null,
    timings: "",
  };
}

describe("deriveAssistantPhase", () => {
  it("prefers offline before every other phase", () => {
    expect(deriveAssistantPhase({
      backendOnline: false,
      profileReady: true,
      isListening: true,
      chatBusy: true,
      isSpeaking: true,
    })).toBe("offline");
  });

  it("shows thinking before speaking or listening", () => {
    expect(deriveAssistantPhase({
      backendOnline: true,
      profileReady: true,
      isListening: true,
      chatBusy: true,
      isSpeaking: true,
    })).toBe("thinking");
  });

  it("falls back to ready when the bot is idle", () => {
    expect(deriveAssistantPhase({
      backendOnline: true,
      profileReady: true,
      isListening: false,
      chatBusy: false,
      isSpeaking: false,
    })).toBe("ready");
  });
});

describe("phaseLabel", () => {
  it("returns user-facing labels", () => {
    expect(phaseLabel("speaking")).toBe("Speaking");
    expect(phaseLabel("needs-profile")).toBe("Choose a profile");
  });
});

describe("transcriptPreview", () => {
  it("prefers the current live transcript when it is meaningful", () => {
    expect(transcriptPreview("I feel a little overwhelmed", [entry("older")]))
      .toBe("I feel a little overwhelmed");
  });

  it("falls back to the latest transcript entry when the live text is placeholder copy", () => {
    expect(transcriptPreview("Listening for harness transcript...", [
      entry("[no transcript text]"),
      entry("Can you help me slow down?"),
    ])).toBe("Can you help me slow down?");
  });

  it("returns a gentle prompt when there is no useful transcript yet", () => {
    expect(transcriptPreview("Mic idle", [])).toBe("Start speaking when you're ready.");
  });
});
