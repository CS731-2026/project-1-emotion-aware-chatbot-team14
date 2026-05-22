<script lang="ts">
  import type { ChatDebug, Profile } from "$lib/api";
  import { EMOTION_EMOJI, EMOTION_LABEL, type Emotion } from "$lib/harness/types";

  type RejectedTranscript = {
    id: string;
    timestamp: string;
    text: string;
    reason: string;
    confidence: number | null;
    durationMs: number | null;
  };

  let {
    profile,
    faceDetected,
    lastDetection,
    latestHarnessFrame,
    latestFaceCrop,
    latestFrameSummary,
    emotion,
    reasoningDebug,
    rejectedTranscripts = [],
    emotionCounts = {},
  }: {
    profile: Profile | null;
    faceDetected: boolean;
    lastDetection: string;
    latestHarnessFrame: string | null;
    latestFaceCrop: string | null;
    latestFrameSummary: string;
    emotion: Emotion;
    reasoningDebug: ChatDebug | null;
    rejectedTranscripts?: RejectedTranscript[];
    emotionCounts?: Record<string, number>;
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

    {#if emotionCounts && Object.keys(emotionCounts).length > 0}
      {@const total = Object.values(emotionCounts).reduce((s, n) => s + n, 0)}
      <article class="debug-panel">
        <p class="panel-label">Session emotion breakdown</p>
        <div class="emotion-stat-bars">
          {#each Object.entries(emotionCounts).sort((a, b) => b[1] - a[1]) as [em, count]}
            {@const pct = Math.round((count / total) * 100)}
            {@const emoji = EMOTION_EMOJI[em as Emotion] ?? "❓"}
            {@const lbl = EMOTION_LABEL[em as Emotion] ?? em}
            <div class="es-row">
              <span class="es-label">{emoji} {lbl}</span>
              <div class="es-track"><div class="es-fill" style="width:{pct}%"></div></div>
              <span class="es-pct">{pct}%</span>
            </div>
          {/each}
        </div>
        <p class="panel-meta">{total} frames analysed</p>
      </article>
    {/if}

    <article class="debug-panel">
      <p class="panel-label">Filtered transcripts</p>
      {#if rejectedTranscripts.length === 0}
        <p class="panel-copy">Nothing filtered yet. Speak quietly or briefly to test.</p>
      {:else}
        <ul class="rejected-list">
          {#each rejectedTranscripts as item (item.id)}
            <li class="rejected-item">
              <div class="rejected-meta">
                <span>{item.timestamp}</span>
                <span>
                  {item.confidence !== null ? `${(item.confidence * 100).toFixed(0)}% conf` : "no conf"}
                  {item.durationMs !== null ? ` · ${(item.durationMs / 1000).toFixed(1)}s` : ""}
                </span>
              </div>
              <p class="rejected-text">{item.text}</p>
              <p class="rejected-reason">{item.reason}</p>
            </li>
          {/each}
        </ul>
      {/if}
    </article>

    <article class="debug-panel">
      <p class="panel-label">Session state</p>
      {#if reasoningDebug?.session_state}
        <div class="session-summary">
          <span class="session-pill" data-surface={reasoningDebug.session_state.surface}>
            {reasoningDebug.session_state.state_name ?? "(none)"}
          </span>
          <span class="session-pill">turn {reasoningDebug.session_state.turn_in_state}</span>
          <span class="session-pill">segment #{reasoningDebug.session_state.segment_id}</span>
        </div>
        {#if reasoningDebug.session_state.emissions.length > 0}
          <p class="panel-heading">Last emissions</p>
          <pre class="debug-pre">{reasoningDebug.session_state.emissions.map((e) => e.name).join(", ")}</pre>
        {/if}
        {#if Object.keys(reasoningDebug.session_state.state_facts).length > 0}
          <p class="panel-heading">State facts</p>
          <pre class="debug-pre">{JSON.stringify(reasoningDebug.session_state.state_facts, null, 2)}</pre>
        {/if}
      {:else}
        <p class="panel-copy">No conductor activity yet. Send a chat turn to start a session.</p>
      {/if}
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

  .session-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.3rem;
  }
  .session-pill {
    border-radius: 999px;
    padding: 0.25rem 0.6rem;
    font-size: 0.78rem;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: rgba(255, 255, 255, 0.88);
  }
  .session-pill[data-surface="checkin"] {
    background: rgba(249, 115, 22, 0.18);
    border-color: rgba(249, 115, 22, 0.36);
  }
  .session-pill[data-surface="done"] {
    background: rgba(74, 222, 128, 0.18);
    border-color: rgba(74, 222, 128, 0.36);
  }

  .rejected-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .rejected-item {
    background: rgba(255, 132, 132, 0.08);
    border: 1px solid rgba(255, 132, 132, 0.18);
    border-radius: 10px;
    padding: 0.55rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .rejected-meta {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    font-size: 0.72rem;
    color: rgba(255, 255, 255, 0.55);
    letter-spacing: 0.04em;
  }
  .rejected-text {
    margin: 0;
    color: rgba(255, 255, 255, 0.92);
    font-size: 0.92rem;
    line-height: 1.4;
  }
  .rejected-reason {
    margin: 0;
    color: rgba(255, 180, 180, 0.78);
    font-size: 0.78rem;
    font-style: italic;
  }

  .emotion-stat-bars { display: flex; flex-direction: column; gap: 0.4rem; }
  .es-row { display: flex; align-items: center; gap: 0.55rem; }
  .es-label { font-size: 0.76rem; color: rgba(255,255,255,0.75); width: 8.5rem; flex-shrink: 0; }
  .es-track { flex: 1; height: 5px; background: rgba(255,255,255,0.1); border-radius: 999px; overflow: hidden; }
  .es-fill { height: 100%; background: rgba(255,215,100,0.65); border-radius: 999px; transition: width 350ms ease; }
  .es-pct { font-size: 0.7rem; color: rgba(255,255,255,0.45); width: 2.2rem; text-align: right; flex-shrink: 0; }

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
