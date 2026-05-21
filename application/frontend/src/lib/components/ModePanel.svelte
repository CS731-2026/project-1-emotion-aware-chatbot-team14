<script lang="ts">
  import { fade, fly, scale } from "svelte/transition";
  import { cubicOut } from "svelte/easing";
  import ChatInput from "./ChatInput.svelte";
  import type { OverlaySpec, OverlayStep } from "$lib/conversation/sampleCheckIns";

  let {
    spec,
    currentStep,
    onAnswer,
    onCancelMic,
    isListening,
    onMicToggle,
  }: {
    spec: OverlaySpec;
    currentStep: number;
    onAnswer: (stepId: string, value: string) => void;
    onCancelMic?: () => void;
    isListening: boolean;
    onMicToggle: () => void;
  } = $props();

  const step = $derived<OverlayStep>(spec.steps[currentStep] ?? spec.steps[0]);
  const totalSteps = $derived(spec.steps.length);
  const showCounter = $derived(totalSteps > 1);
  const showChatInput = $derived(
    spec.captureMode === "conversational" && (spec.allowFreeText ?? true)
  );

  function recordAnswer(value: string) {
    // Cancel any half-spoken utterance so we don't double-send alongside the chip.
    if (spec.captureMode === "conversational") onCancelMic?.();
    onAnswer(step.id, value);
  }
</script>

<div
  class="panel"
  in:scale={{ start: 0.94, duration: 320, delay: 280, easing: cubicOut }}
  out:scale={{ start: 1, duration: 220, easing: cubicOut }}
>
  <p class="kicker" in:fade={{ duration: 240, delay: 360 }}>{spec.kicker}</p>

  {#key step.id}
    <h2 class="prompt" in:fade={{ duration: 240, delay: 60 }}>{step.prompt}</h2>
    {#if step.subtext}
      <p class="subtext" in:fade={{ duration: 240, delay: 100 }}>{step.subtext}</p>
    {/if}

    <div class="choices">
      {#each step.choices as choice, i (choice.value)}
        <button
          class="choice"
          data-tone={choice.tone ?? "neutral"}
          onclick={() => recordAnswer(choice.value)}
          in:fly={{ y: 12, duration: 240, delay: 140 + i * 40, easing: cubicOut }}
          type="button"
        >
          {choice.label}
        </button>
      {/each}
    </div>
  {/key}

  {#if showChatInput}
    <div class="composer" in:fade={{ duration: 240, delay: 500 }}>
      <ChatInput onSend={recordAnswer} {isListening} {onMicToggle} />
    </div>
  {/if}

  {#if showCounter}
    <p class="step" in:fade={{ duration: 240, delay: 600 }}>
      Step {currentStep + 1} of {totalSteps}
    </p>
  {/if}
</div>

<style>
  .panel {
    width: min(560px, 100%);
    padding: 1.75rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
    align-items: center;
    text-align: center;
    border-radius: 24px;
    background: rgba(15, 18, 28, 0.78);
    border: 1px solid rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(20px);
    color: #f5f7fb;
    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.4);
  }
  .kicker {
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.72rem;
    color: rgba(255, 255, 255, 0.62);
    margin: 0;
  }
  .prompt {
    font-size: clamp(1.4rem, 3vw, 1.9rem);
    line-height: 1.2;
    margin: 0;
  }
  .subtext {
    color: rgba(255, 255, 255, 0.7);
    margin: 0;
  }
  .choices {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 0.4rem;
  }
  .choice {
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 999px;
    padding: 0.6rem 1.05rem;
    background: rgba(255, 255, 255, 0.06);
    color: #f5f7fb;
    cursor: pointer;
    font-size: 0.95rem;
    transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
  }
  .choice:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 255, 255, 0.28);
  }
  .choice:active {
    transform: scale(0.97);
  }
  .choice[data-tone="positive"] {
    border-color: rgba(74, 222, 128, 0.45);
    background: rgba(74, 222, 128, 0.14);
  }
  .choice[data-tone="concerning"] {
    border-color: rgba(255, 132, 132, 0.4);
    background: rgba(255, 82, 82, 0.12);
  }
  .composer {
    width: 100%;
    margin-top: 0.5rem;
  }
  .step {
    font-size: 0.78rem;
    color: rgba(255, 255, 255, 0.5);
    margin: 0.5rem 0 0;
  }
</style>
