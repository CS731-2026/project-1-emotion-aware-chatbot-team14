<script lang="ts">
  import type { TranscriptEntry } from "$lib/harness/types";

  let {
    entries,
    compact = false,
  }: {
    entries: TranscriptEntry[];
    compact?: boolean;
  } = $props();
</script>

<div class:compact class="transcript-log">
  <p class="card-label">Speech transcript history</p>
  {#if entries.length}
    {#each entries as entry}
      <article class:error={entry.error}>
        <span>{entry.timestamp}{entry.chunk ? ` · chunk ${entry.chunk}` : ""}</span>
        <strong>{entry.text}</strong>
        {#if entry.timings}
          <small>{entry.timings}</small>
        {/if}
        {#if entry.error}
          <small>{entry.error}</small>
        {/if}
      </article>
    {/each}
  {:else}
    <p>No transcript returned yet.</p>
  {/if}
</div>

<style>
  .transcript-log {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
  }

  article {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 0.65rem 0.7rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.055);
  }

  article.error {
    border-color: rgba(248, 113, 113, 0.35);
    background: rgba(127, 29, 29, 0.2);
  }

  span,
  small {
    color: rgba(255, 255, 255, 0.62);
    font-size: 0.72rem;
    line-height: 1.3;
  }

  strong {
    color: rgba(255, 255, 255, 0.92);
    line-height: 1.35;
    word-break: break-word;
  }

  .compact article {
    padding: 0.55rem 0.6rem;
  }
</style>
