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
    placeholder="Type a message..."
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
    {isListening ? "Listening" : "Mic"}
  </button>
  <button class="send-btn" onclick={submit} {disabled} type="button">Send</button>
</div>

<style>
  .input-bar {
    display: flex;
    align-items: flex-end;
    gap: 0.5rem;
    padding-top: 0.85rem;
    border-top: 1px solid var(--color-border);
  }

  textarea {
    flex: 1;
    resize: none;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    font-size: 0.95rem;
    padding: 0.7rem 0.85rem;
    min-height: 2.75rem;
    max-height: 8rem;
    overflow-y: auto;
    line-height: 1.4;
  }

  .mic-btn,
  .send-btn {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    cursor: pointer;
    font-size: 0.9rem;
    padding: 0.7rem 0.9rem;
    white-space: nowrap;
  }

  .mic-btn:hover,
  .send-btn:hover {
    background: var(--color-surface-hover);
  }

  .send-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .mic-btn.active {
    border-color: rgba(255, 80, 80, 0.55);
    background: rgba(255, 60, 60, 0.15);
  }
</style>
