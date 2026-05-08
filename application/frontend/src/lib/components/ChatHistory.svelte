<script lang="ts">
  import type { Message } from "$lib/api";

  let { messages }: { messages: Message[] } = $props();

  let scrollEl = $state<HTMLDivElement | undefined>(undefined);

  $effect(() => {
    void messages.length;
    if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
  });

  function formatTime(ts: string) {
    return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
</script>

<div class="history" bind:this={scrollEl}>
  {#each messages as msg (msg.id)}
    <div class="bubble-wrap" class:user={msg.role === "user"} class:agent={msg.role === "agent"}>
      <div class="bubble">
        <p class="text">{msg.content}</p>
        <span class="time">{formatTime(msg.timestamp)}</span>
      </div>
    </div>
  {/each}
</div>

<style>
  .history {
    flex: 1;
    overflow-y: auto;
    padding: 0.5rem 0;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.2) transparent;
    min-height: 0;
  }

  .bubble-wrap {
    display: flex;
  }

  .bubble-wrap.user {
    justify-content: flex-end;
  }

  .bubble-wrap.agent {
    justify-content: flex-start;
  }

  .bubble {
    max-width: 72%;
    padding: 0.75rem 0.95rem;
    border-radius: var(--radius);
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }

  .user .bubble {
    background: var(--color-user-bubble);
    border-bottom-right-radius: 4px;
  }

  .agent .bubble {
    background: var(--color-agent-bubble);
    border-bottom-left-radius: 4px;
  }

  .text {
    font-size: 0.95rem;
    line-height: 1.45;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .time {
    font-size: 0.72rem;
    color: var(--color-text-muted);
    align-self: flex-end;
  }
</style>
