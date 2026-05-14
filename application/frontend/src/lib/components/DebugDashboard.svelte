<script lang="ts">
  import type { ChatDebug, Profile } from "$lib/api";
  import type { Emotion } from "$lib/harness/types";

  let {
    profile,
    faceDetected,
    lastDetection,
    latestHarnessFrame,
    latestFaceCrop,
    latestFrameSummary,
    emotion,
    reasoningDebug,
  }: {
    profile: Profile | null;
    faceDetected: boolean;
    lastDetection: string;
    latestHarnessFrame: string | null;
    latestFaceCrop: string | null;
    latestFrameSummary: string;
    emotion: Emotion;
    reasoningDebug: ChatDebug | null;
  } = $props();
</script>

<div class="debug-shell">
  <div class="debug-topbar">
    <div>
      <p class="debug-label">Debug view</p>
      <h1>Face + reasoning trace</h1>
    </div>
    <div class="debug-profile">{profile?.name ?? "No profile selected"}</div>
  </div>

  <section class="debug-panels">
    <article class="debug-panel">
      <p class="panel-label">Latest face</p>
      <div class="emotion-row">
        <span class:detected={faceDetected} class="emotion-pill">
          {faceDetected ? "Face detected" : "No face"}
        </span>
        <span class="emotion-pill active">{emotion}</span>
      </div>
      <p class="panel-copy">{lastDetection}</p>

      <div class="frame-grid">
        {#if latestHarnessFrame}
          <img class="debug-frame" src={latestHarnessFrame} alt="Latest harness frame" />
        {/if}
        {#if latestFaceCrop}
          <img class="debug-frame face-crop" src={latestFaceCrop} alt="Latest detected face crop" />
        {/if}
      </div>

      <pre class="debug-pre">{latestFrameSummary}</pre>
    </article>

    <article class="debug-panel">
      <p class="panel-label">Reasoning transcript</p>

      {#if reasoningDebug}
        <p class="panel-meta">{reasoningDebug.provider ?? "no-provider"} / {reasoningDebug.model ?? "no-model"}</p>

        <p class="panel-heading">Current message</p>
        <pre class="debug-pre">{reasoningDebug.current_message}</pre>

        <p class="panel-heading">Emotional context</p>
        <pre class="debug-pre">{reasoningDebug.emotional_context}</pre>

        <p class="panel-heading">Transcript context</p>
        <pre class="debug-pre">{reasoningDebug.transcript_lines.length ? reasoningDebug.transcript_lines.join("\n") : "[no transcript context]"}</pre>
      {:else}
        <p class="panel-copy">Send a message to populate the latest reasoning trace.</p>
      {/if}
    </article>
  </section>
</div>

<style>
  .debug-shell {
    position: fixed;
    right: 1rem;
    top: 1rem;
    width: min(640px, calc(100vw - 2rem));
    max-height: calc(100vh - 2rem);
    overflow: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
    z-index: 20;
    background: rgba(5, 8, 16, 0.78);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    backdrop-filter: blur(24px);
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.35);
  }

  .debug-topbar {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
  }

  .debug-label,
  .panel-label,
  .panel-heading {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: rgba(255, 255, 255, 0.62);
  }

  .debug-topbar h1 {
    font-size: clamp(1.5rem, 4vw, 2.2rem);
    margin: 0.2rem 0;
  }

  .debug-profile,
  .debug-panel {
    background: rgba(6, 10, 20, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    backdrop-filter: blur(12px);
  }

  .debug-profile {
    padding: 0.85rem 1rem;
    color: rgba(255, 255, 255, 0.9);
  }

  .debug-panels {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.85rem;
  }

  .debug-panel {
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
    min-height: 0;
  }

  .emotion-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .emotion-pill {
    border-radius: 999px;
    padding: 0.4rem 0.7rem;
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.82);
    font-size: 0.85rem;
  }

  .emotion-pill.detected {
    border-color: rgba(74, 222, 128, 0.35);
    background: rgba(34, 197, 94, 0.13);
  }

  .emotion-pill.active {
    border-color: rgba(255, 215, 128, 0.35);
    background: rgba(255, 214, 10, 0.12);
    text-transform: capitalize;
  }

  .panel-copy,
  .panel-meta {
    color: rgba(255, 255, 255, 0.8);
    line-height: 1.45;
  }

  .debug-frame {
    width: 100%;
    max-height: 180px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    object-fit: cover;
  }

  .face-crop {
    border-color: rgba(74, 222, 128, 0.45);
  }

  .frame-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.65rem;
  }

  .debug-pre {
    white-space: pre-wrap;
    word-break: break-word;
    color: rgba(255, 255, 255, 0.88);
    line-height: 1.5;
    font-size: 0.86rem;
  }

  @media (max-width: 720px) {
    .debug-shell {
      padding: 1rem;
      top: auto;
      bottom: 1rem;
      right: 1rem;
      width: min(100vw - 1rem, 100%);
      max-height: 62vh;
    }

    .debug-topbar {
      flex-direction: column;
    }

    .frame-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
