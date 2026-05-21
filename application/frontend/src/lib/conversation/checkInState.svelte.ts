import type { CheckInSpec } from "./sampleCheckIns";

export type { CheckInSpec, OverlaySpec, PageSpec, QuestionSpec, OverlayStep, Choice } from "./sampleCheckIns";

type CheckInSource = "debug" | "mode";

type CheckInStore = {
  active: boolean;
  spec: CheckInSpec | null;
  source: CheckInSource | null;
  // Index into spec.steps when spec.elevation === "overlay" with multiple steps.
  // Always 0 at open; advanced via advanceOverlayStep().
  currentStep: number;
};

export const checkInState = $state<CheckInStore>({
  active: false,
  spec: null,
  source: null,
  currentStep: 0,
});

export function openCheckIn(spec: CheckInSpec, source: CheckInSource): void {
  checkInState.spec = spec;
  checkInState.source = source;
  checkInState.currentStep = 0;
  checkInState.active = true;
}

export function closeCheckIn(): void {
  // Spec stays in place so leave-transitions can render against it; replaced
  // when the next check-in opens.
  checkInState.active = false;
}

/**
 * Advance to the next overlay step. Returns true if a next step exists,
 * false if we just answered the last step (caller should close).
 * No-op (returns false) for non-overlay or non-active states.
 */
export function advanceOverlayStep(): boolean {
  const spec = checkInState.spec;
  if (!checkInState.active || spec?.elevation !== "overlay") return false;
  const next = checkInState.currentStep + 1;
  if (next >= spec.steps.length) return false;
  checkInState.currentStep = next;
  return true;
}
