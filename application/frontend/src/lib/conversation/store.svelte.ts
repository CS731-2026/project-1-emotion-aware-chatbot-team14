import type { Mode, Stage } from "$lib/api";

// Mirrors model_service FeedbackEvent. Populated when Phase 4 wires up
// the /feedback route; until then this list stays empty.
export type FeedbackEvent = {
  timestamp: number;
  reported: string;
  predicted: string;
  predictedConfidence: number;
  mismatch: boolean;
  stage: Stage | null;
};

type ConversationState = {
  mode: Mode;
  stage: Stage | null;
  // Number of user turns spent in the current qa stage. Reset on stage change.
  qaTurnCount: number;
  // 0-indexed count of self-report ticks completed in the current feedback visit.
  feedbackTickIndex: number;
  // Client-side mirror of recent feedback events for UI display. Authoritative
  // copy lives in the model_service session buffer + backend jsonl log.
  feedbackEvents: FeedbackEvent[];
};

export const conversationState = $state<ConversationState>({
  mode: "qa",
  stage: null,
  qaTurnCount: 0,
  feedbackTickIndex: 0,
  feedbackEvents: [],
});

export function setMode(mode: Mode): void {
  conversationState.mode = mode;
}

export function setStage(stage: Stage | null): void {
  conversationState.stage = stage;
  conversationState.qaTurnCount = 0;
}

export function recordQaTurn(): void {
  conversationState.qaTurnCount += 1;
}

export function resetFeedbackTicks(): void {
  conversationState.feedbackTickIndex = 0;
}

export function recordFeedbackTick(event: FeedbackEvent): void {
  conversationState.feedbackTickIndex += 1;
  conversationState.feedbackEvents = [...conversationState.feedbackEvents, event];
}
