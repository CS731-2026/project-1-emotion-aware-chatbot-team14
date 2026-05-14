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
    placeholder="Type if you'd rather not speak..."
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
    {isListening ? "Listening" : "Speak"}
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
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
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
