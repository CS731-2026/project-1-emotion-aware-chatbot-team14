// Spec types for the check-in surface plus debug fixtures for Shift+1..4.
//
// The reasoner is not wired yet. For elevation-2 pages, a question may declare
// `cannedReactionOn` so we can validate the inline reaction UI without a real
// model roundtrip.

export type Choice = {
  label: string;
  value: string;
  tone?: "neutral" | "positive" | "concerning";
};

export type CannedReaction = {
  value: string;
  alert?: string;
  assistant?: string;
};

export type QuestionSpec = {
  id: string;
  prompt: string;
  choices: Choice[];
  allowFreeText?: boolean;
  cannedReactionOn?: CannedReaction;
};

// One step inside an overlay check-in. A single-step overlay (e.g. consent
// branch picker) is just an OverlaySpec with steps.length === 1.
export type OverlayStep = {
  id: string;
  prompt: string;
  subtext?: string;
  choices: Choice[];
};

export type OverlaySpec = {
  elevation: "overlay";
  kicker: string;
  steps: OverlayStep[];
  captureMode: "conversational" | "static";
  allowFreeText?: boolean;
};

export type PageSpec = {
  elevation: "page";
  title: string;
  subtitle?: string;
  emotionAware?: boolean;
  questions: QuestionSpec[];
  reveal: "all-at-once" | "sequential";
};

export type CheckInSpec = OverlaySpec | PageSpec;

// ---------- Debug fixtures ----------

export const SAMPLE_OVERLAY_CONVERSATIONAL: OverlaySpec = {
  elevation: "overlay",
  kicker: "A quick check-in",
  captureMode: "conversational",
  allowFreeText: true,
  steps: [
    {
      id: "visit_overall",
      prompt: "How would you say your visit went today?",
      choices: [
        { label: "It went well", value: "It went well", tone: "positive" },
        { label: "It was okay", value: "It was okay" },
        { label: "It was rough", value: "It was rough", tone: "concerning" },
      ],
    },
    {
      id: "ai_explained",
      prompt: "Did the doctor explain what the AI was used for in a way you understood?",
      choices: [
        { label: "Yes, clearly", value: "Yes, clearly", tone: "positive" },
        { label: "Sort of", value: "Sort of" },
        { label: "Not really", value: "Not really", tone: "concerning" },
      ],
    },
    {
      id: "ai_comfort",
      prompt: "How did you feel about a computer being involved in your care?",
      choices: [
        { label: "Comfortable", value: "Comfortable", tone: "positive" },
        { label: "Neutral", value: "Neutral" },
        { label: "Uncomfortable", value: "Uncomfortable", tone: "concerning" },
      ],
    },
    {
      id: "still_worried",
      prompt: "Is there anything you're still worried about?",
      choices: [
        { label: "No, I'm good", value: "No, I'm good", tone: "positive" },
        { label: "A small thing", value: "A small thing" },
        { label: "Yes, quite a bit", value: "Yes, quite a bit", tone: "concerning" },
      ],
    },
  ],
};

export const SAMPLE_OVERLAY_STATIC: OverlaySpec = {
  elevation: "overlay",
  kicker: "Before we begin",
  captureMode: "static",
  steps: [
    {
      id: "intent",
      prompt: "What brings you here today?",
      choices: [
        { label: "I want to ask a question", value: "ask" },
        { label: "I want to give feedback", value: "feedback" },
      ],
    },
  ],
};

export const SAMPLE_PAGE_SEQUENTIAL: PageSpec = {
  elevation: "page",
  title: "A few quick questions",
  subtitle: "Take your time — there are no right or wrong answers.",
  emotionAware: true,
  reveal: "sequential",
  questions: [
    {
      id: "doctor_listened",
      prompt: "Did you feel your doctor listened to you during today's visit?",
      choices: [
        { label: "Yes, completely", value: "Yes, completely", tone: "positive" },
        { label: "Mostly", value: "Mostly" },
        { label: "Not really", value: "Not really" },
        { label: "No", value: "No", tone: "concerning" },
      ],
    },
    {
      id: "ai_comfort",
      prompt: "How do you feel about the computer or AI being used during your appointment?",
      choices: [
        { label: "Very comfortable", value: "Very comfortable", tone: "positive" },
        { label: "I feel fine", value: "I feel fine" },
        { label: "A bit unsure", value: "A bit unsure" },
        { label: "Worried", value: "Worried", tone: "concerning" },
      ],
      cannedReactionOn: {
        value: "I feel fine",
        alert:
          'We noticed you selected "I feel fine" — but it seems you might have some concerns. ' +
          "That's completely okay. Would you like us to explain what the AI actually did during your visit?",
        assistant:
          "The AI helped your doctor by suggesting possible causes based on your symptoms — " +
          "your doctor made all the final decisions. You are always in control. " +
          "Would you like to know more, or ask a question?",
      },
    },
    {
      id: "still_worried",
      prompt: "Is there anything you're still worried about?",
      choices: [
        { label: "No, I'm good", value: "No, I'm good", tone: "positive" },
        { label: "A small thing", value: "A small thing" },
        { label: "Yes, quite a bit", value: "Yes, quite a bit", tone: "concerning" },
      ],
      allowFreeText: true,
    },
  ],
};

export const SAMPLE_PAGE_ALL_AT_ONCE: PageSpec = {
  ...SAMPLE_PAGE_SEQUENTIAL,
  reveal: "all-at-once",
};
