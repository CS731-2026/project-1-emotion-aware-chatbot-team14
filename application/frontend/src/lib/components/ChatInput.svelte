<script lang="ts">
  let {
    onSend,
    isListening,
    onMicToggle,
    disabled = false,
    placeholder = "Type if you'd rather not speak...",
    audioLevel = 0,
    locked = false,
  }: {
    onSend: (text: string) => void;
    isListening: boolean;
    onMicToggle: () => void;
    disabled?: boolean;
    placeholder?: string;
    /** Current RMS audio level from the VAD, normalised 0–1. Drives the
     * pulse animation on the listening button. */
    audioLevel?: number;
    /** Mic is gated (assistant is replying / speaking). Surfaces the same
     * "can't hear you right now" state the recording-subtitle uses. */
    locked?: boolean;
  } = $props();

  // Map raw RMS (typically 0–0.02 in normal speech) to a 0–1 pulse value.
  const micPulse = $derived(Math.min(1, Math.max(0, audioLevel / 0.012)));

  let text = $state("");

  function submit() {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    text = "";
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }
</script>

<div class="input-bar">
  <textarea
    bind:value={text}
    onkeydown={handleKeydown}
    {placeholder}
    rows="1"
    {disabled}
  ></textarea>
  <button
    class="mic-btn"
    class:active={isListening}
    class:locked
    style={`--mic-pulse:${micPulse};`}
    onclick={onMicToggle}
    title={isListening ? "Stop mic" : "Start mic"}
    type="button"
  >
    <span class="mic-bars" aria-hidden="true">
      <span></span><span></span><span></span><span></span>
    </span>
    <span class="mic-label">
      {#if locked}
        Listening (paused)
      {:else if isListening}
        Listening
      {:else}
        Speak
      {/if}
    </span>
  </button>
  <button class="send-btn" onclick={submit} {disabled} type="button">Reply</button>
</div>

<style>
  .input-bar {
    display: flex;
    align-items: flex-end;
    gap: 0.65rem;
    width: min(760px, 100%);
    padding: 0.9rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    background: rgba(9, 13, 24, 0.28);
    backdrop-filter: blur(18px);
  }

  textarea {
    flex: 1;
    resize: none;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    font-size: 0.95rem;
    padding: 0.85rem 1rem;
    min-height: 3rem;
    max-height: 8rem;
    overflow-y: auto;
    line-height: 1.4;
  }

  .mic-btn,
  .send-btn {
    border-radius: 999px;
    cursor: pointer;
    font-size: 0.9rem;
    padding: 0.85rem 1rem;
    white-space: nowrap;
  }

  .mic-btn {
    --mic-pulse: 0;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    transition: background 140ms ease, border-color 140ms ease;
  }
  .mic-bars {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    height: 1rem;
  }
  .mic-bars span {
    display: block;
    width: 3px;
    border-radius: 2px;
    background: rgba(255, 255, 255, 0.35);
    height: calc(15% + var(--mic-pulse) * 70%);
    transition: height 90ms ease, background 140ms ease;
  }
  .mic-btn.active .mic-bars span {
    background: rgb(255, 108, 108);
  }
  /* The four bars dance at slightly different magnitudes so the cluster
     reads as "live audio", not "synchronised pulse". */
  .mic-bars span:nth-child(1) { height: calc(25% + var(--mic-pulse) * 55%); }
  .mic-bars span:nth-child(2) { height: calc(35% + var(--mic-pulse) * 65%); animation-delay: 60ms; }
  .mic-bars span:nth-child(3) { height: calc(45% + var(--mic-pulse) * 55%); animation-delay: 120ms; }
  .mic-bars span:nth-child(4) { height: calc(20% + var(--mic-pulse) * 60%); animation-delay: 180ms; }

  .mic-btn.locked {
    border-color: rgba(168, 85, 247, 0.55);
    background: rgba(124, 58, 237, 0.18);
    color: rgba(233, 213, 255, 0.92);
  }
  .mic-btn.locked .mic-bars span {
    background: rgba(216, 180, 254, 0.65);
  }

  .send-btn {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.35);
    color: #09111d;
    font-weight: 600;
  }

  .mic-btn:hover,
  .send-btn:hover {
    background: var(--color-surface-hover);
  }

  .send-btn:hover {
    background: white;
  }

  .send-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .mic-btn.active {
    border-color: rgba(255, 80, 80, 0.55);
    background: rgba(255, 60, 60, 0.15);
  }

  @media (max-width: 720px) {
    .input-bar {
      width: 100%;
      padding: 0.75rem;
      gap: 0.5rem;
    }

    .mic-btn,
    .send-btn {
      padding: 0.8rem 0.9rem;
    }
  }
</style>
