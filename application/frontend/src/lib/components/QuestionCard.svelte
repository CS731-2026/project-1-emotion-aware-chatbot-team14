<script lang="ts">
  import type { QuestionSpec } from "$lib/conversation/sampleCheckIns";
  import InlineAlert from "./InlineAlert.svelte";
  import AssistantBubble from "./AssistantBubble.svelte";

  export type ReactionEntry =
    | { kind: "alert"; text: string }
    | { kind: "assistant"; text: string };

  let {
    question,
    selectedValue,
    reactions,
    onSelect,
    disabled = false,
  }: {
    question: QuestionSpec;
    selectedValue: string | null;
    reactions: ReactionEntry[];
    onSelect: (value: string) => void;
    disabled?: boolean;
  } = $props();
</script>

<section class="card">
  <p class="prompt">{question.prompt}</p>

  <div class="choices">
    {#each question.choices as choice (choice.value)}
      <button
        class="choice"
        data-tone={choice.tone ?? "neutral"}
        class:selected={selectedValue === choice.value}
        {disabled}
        onclick={() => onSelect(choice.value)}
        type="button"
      >
        {choice.label}
      </button>
    {/each}
  </div>

  {#if reactions.length > 0}
    <div class="reactions">
      {#each reactions as reaction, i (i)}
        {#if reaction.kind === "alert"}
          <InlineAlert text={reaction.text} />
        {:else}
          <AssistantBubble text={reaction.text} />
        {/if}
      {/each}
    </div>
  {/if}
</section>

<style>
  .card {
    padding: 1.5rem 1.4rem;
    background: #f5f5f4;
    border-radius: 16px;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .prompt {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
    line-height: 1.35;
    color: #1c1917;
  }
  .choices {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
  }
  .choice {
    border: 1px solid #d6d3d1;
    border-radius: 999px;
    padding: 0.55rem 1.1rem;
    background: #ffffff;
    color: #1c1917;
    font-size: 0.95rem;
    cursor: pointer;
    transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
  }
  .choice:hover:not(:disabled) {
    background: #fafaf9;
    border-color: #a8a29e;
  }
  .choice:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .choice.selected[data-tone="positive"] {
    background: rgba(187, 247, 208, 0.65);
    border-color: rgba(34, 197, 94, 0.5);
    color: #14532d;
  }
  /* Both neutral and concerning render in amber when selected so a "neutral"
     answer that triggers an inline reaction still reads as attention-worthy. */
  .choice.selected[data-tone="neutral"],
  .choice.selected[data-tone="concerning"] {
    background: rgba(254, 215, 170, 0.55);
    border-color: rgba(249, 115, 22, 0.5);
    color: #7c2d12;
  }
  .reactions {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
</style>
