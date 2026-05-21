<script lang="ts">
  import { fade, fly } from "svelte/transition";
  import { cubicOut } from "svelte/easing";
  import ChatInput from "./ChatInput.svelte";
  import QuestionCard, { type ReactionEntry } from "./QuestionCard.svelte";
  import type { PageSpec, QuestionSpec } from "$lib/conversation/sampleCheckIns";

  type AnswerState = {
    selected: string | null;
    reactions: ReactionEntry[];
  };

  let {
    spec,
    onAnswer,
    onTextSubmit,
    onCancelMic,
    isListening,
    onMicToggle,
    freeTextNote = "",
  }: {
    spec: PageSpec;
    /**
     * Fired whenever the user picks a chip. `isLastAnswer` is true when
     * this answer brings every question in the spec to a selected state —
     * the parent uses it to signal form_complete to the conductor.
     */
    onAnswer: (questionId: string, value: string, isLastAnswer: boolean) => void;
    onTextSubmit: (text: string) => void;
    onCancelMic?: () => void;
    isListening: boolean;
    onMicToggle: () => void;
    /** Most recent free-text supplementary note (typed or spoken). Shown as
     * an acknowledgement under the composer so the user knows it landed. */
    freeTextNote?: string;
  } = $props();

  // Per-question local state. Resets when a different spec object arrives.
  let answers = $state<Record<string, AnswerState>>({});
  // Tracks which question prompts have already been spoken so the same
  // utterance isn't fired again on every reactive rerender. Resets when a
  // new spec arrives.
  let spokenQuestionIds = $state<Set<string>>(new Set());
  $effect(() => {
    spec;
    answers = {};
    spokenQuestionIds = new Set();
  });

  function speakPrompt(text: string) {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    // Cancel anything that was mid-utterance — we always want the newest
    // question to win.
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const enVoice = window.speechSynthesis.getVoices()
      .find((v) => v.lang.toLowerCase().startsWith("en"));
    if (enVoice) utterance.voice = enVoice;
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.volume = 1;
    window.speechSynthesis.speak(utterance);
  }

  const visibleQuestions = $derived.by(() => {
    if (spec.reveal === "all-at-once") return spec.questions;
    // Sequential: reveal up to and including the first unanswered question.
    const out: QuestionSpec[] = [];
    for (const q of spec.questions) {
      out.push(q);
      if (!answers[q.id]?.selected) break;
    }
    return out;
  });

  const progress = $derived(
    spec.questions.filter((q) => answers[q.id]?.selected).length / spec.questions.length
  );

  // Speak each newly-revealed prompt once. Sequential reveal naturally drives
  // this: as the user answers a question the next one appears and we read it.
  $effect(() => {
    for (const q of visibleQuestions) {
      if (!spokenQuestionIds.has(q.id)) {
        speakPrompt(q.prompt);
        spokenQuestionIds = new Set([...spokenQuestionIds, q.id]);
        break;  // one new prompt per tick — don't stack utterances
      }
    }
  });

  function selectChoice(question: QuestionSpec, value: string) {
    onCancelMic?.();
    const reactions: ReactionEntry[] = [];
    const canned = question.cannedReactionOn;
    if (canned && canned.value === value) {
      if (canned.alert) reactions.push({ kind: "alert", text: canned.alert });
      if (canned.assistant) reactions.push({ kind: "assistant", text: canned.assistant });
    }
    const updatedAnswers = { ...answers, [question.id]: { selected: value, reactions } };
    answers = updatedAnswers;
    const allAnswered = spec.questions.every((q) => updatedAnswers[q.id]?.selected);
    onAnswer(question.id, value, allAnswered);
  }
</script>

<div class="page" in:fly={{ y: 24, duration: 360, delay: 280, easing: cubicOut }}>
  <header class="header" in:fade={{ duration: 240, delay: 400 }}>
    <div class="eyebrow">SCREEN 2 — QUESTIONNAIRE WITH EMOTION DETECTION</div>
    <div class="title-row">
      <div class="title-block">
        <h1>{spec.title}</h1>
        {#if spec.subtitle}<p class="subtitle">{spec.subtitle}</p>{/if}
      </div>
      {#if spec.emotionAware}
        <span class="emotion-pill">
          <span aria-hidden="true">☺</span> Emotion-aware
        </span>
      {/if}
    </div>
    <div class="progress" aria-hidden="true">
      <div class="progress-fill" style="width: {progress * 100}%"></div>
    </div>
  </header>

  <div class="questions">
    {#each visibleQuestions as question, i (question.id)}
      <div in:fly={{ y: 16, duration: 280, delay: 500 + i * 80, easing: cubicOut }}>
        <QuestionCard
          {question}
          selectedValue={answers[question.id]?.selected ?? null}
          reactions={answers[question.id]?.reactions ?? []}
          onSelect={(value) => selectChoice(question, value)}
        />
      </div>
    {/each}
  </div>

  <footer class="composer" in:fade={{ duration: 240, delay: 700 }}>
    <ChatInput
      onSend={onTextSubmit}
      {isListening}
      {onMicToggle}
      placeholder="Add anything else you'd like to share (optional)"
    />
    {#if freeTextNote}
      <p class="free-text-ack" aria-live="polite">Got it: &ldquo;{freeTextNote}&rdquo;</p>
    {/if}
  </footer>
</div>

<style>
  .page {
    width: min(960px, 100%);
    margin: 0 auto;
    padding: 2rem;
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 20px 80px rgba(0, 0, 0, 0.18);
    display: flex;
    flex-direction: column;
    gap: 1.4rem;
    color: #1c1917;
  }
  .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: #78716c;
    margin-bottom: 1rem;
  }
  .title-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
  }
  h1 {
    margin: 0 0 0.3rem;
    font-size: 1.5rem;
  }
  .subtitle {
    margin: 0;
    color: #57534e;
    font-size: 1rem;
  }
  .emotion-pill {
    background: rgba(187, 247, 208, 0.5);
    color: #166534;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    font-size: 0.85rem;
    white-space: nowrap;
    border: 1px solid rgba(34, 197, 94, 0.25);
  }
  .progress {
    margin-top: 0.9rem;
    height: 4px;
    background: #e7e5e4;
    border-radius: 2px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: #4f46e5;
    transition: width 320ms cubic-bezier(0.22, 1, 0.36, 1);
  }
  .questions {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .composer {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
    padding-top: 0.5rem;
  }
  .free-text-ack {
    margin: 0;
    font-size: 0.85rem;
    color: #4f46e5;
    font-style: italic;
  }
  @media (max-width: 720px) {
    .page {
      padding: 1.2rem;
      border-radius: 12px;
    }
    .title-row {
      flex-direction: column;
      align-items: flex-start;
    }
  }
</style>
