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
  } = $props();

  // Per-question local state. Resets when a different spec object arrives.
  let answers = $state<Record<string, AnswerState>>({});
  $effect(() => {
    spec;
    answers = {};
  });

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
    <ChatInput onSend={onTextSubmit} {isListening} {onMicToggle} />
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
    justify-content: center;
    padding-top: 0.5rem;
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
