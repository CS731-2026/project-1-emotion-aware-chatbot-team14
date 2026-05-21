import type { CheckInSpec } from "./sampleCheckIns";

export type { CheckInSpec, OverlaySpec, PageSpec, QuestionSpec, Choice } from "./sampleCheckIns";

type CheckInSource = "debug" | "mode";

type CheckInStore = {
  active: boolean;
  spec: CheckInSpec | null;
  source: CheckInSource | null;
};

export const checkInState = $state<CheckInStore>({
  active: false,
  spec: null,
  source: null,
});

export function openCheckIn(spec: CheckInSpec, source: CheckInSource): void {
  checkInState.spec = spec;
  checkInState.source = source;
  checkInState.active = true;
}

export function closeCheckIn(): void {
  // Spec stays in place so leave-transitions can render against it; it gets
  // replaced when the next check-in opens.
  checkInState.active = false;
}
