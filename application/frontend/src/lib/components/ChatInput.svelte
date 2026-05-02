<script lang="ts">
  let {
    onSend,
    isListening,
    onMicToggle,
    disabled = false,
  }: {
    onSend: (text: string) => void;
    isListening: boolean;
    onMicToggle: () => void;
    disabled?: boolean;
  } = $props();

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
    placeholder="Type a message…"
    rows="1"
    {disabled}
  ></textarea>
  <button
    class="mic-btn"
    class:active={isListening}
    onclick={onMicToggle}
    title={isListening ? "Stop mic" : "Start mic"}
    type="button"
  >
    {isListening ? "🔴" : "🎤"}
  </button>
  <button class="send-btn" onclick={submit} {disabled} type="button">Send</button>
</div>

<style>
  .input-bar {
    display: flex;
    align-items: flex-end;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    border-top: 1px solid var(--color-border);
    background: rgba(0, 0, 0, 0.2);
  }

  textarea {
    flex: 1;
    resize: none;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    color: var(--color-text);
    font-family: var(--font);
    font-size: 0.95rem;
    padding: 0.6rem 0.8rem;
    min-height: 2.5rem;
    max-height: 8rem;
    overflow-y: auto;
    outline: none;
    line-height: 1.4;
  }

  textarea:focus {
    border-color: rgba(255, 255, 255, 0.3);
  }

  .mic-btn, .send-btn {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    color: var(--color-text);
    cursor: pointer;
    font-size: 0.9rem;
    padding: 0.55rem 0.85rem;
    white-space: nowrap;
    transition: background 0.15s;
  }

  .mic-btn:hover, .send-btn:hover {
    background: var(--color-surface-hover);
  }

  .send-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .mic-btn.active {
    border-color: rgba(255, 80, 80, 0.6);
    background: rgba(255, 60, 60, 0.15);
  }
</style>
