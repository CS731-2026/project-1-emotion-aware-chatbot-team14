<script lang="ts">
  import type { ChatDebug } from "$lib/api";
  import { SPEECH_THRESHOLD, type TranscriptEntry } from "$lib/harness/types";
  import TranscriptHistory from "$lib/components/TranscriptHistory.svelte";

  let {
    harnessStatus,
    websocketDebug,
    websocketEvents,
    liveTranscript,
    vadState,
    currentAudioLevel,
    audioDebugEvents,
    transcriptEntries,
    lastDetection,
    harnessFrameCount,
    latestHarnessFrame,
    latestFaceCrop,
    latestFrameSummary,
    harnessAudioCount,
    latestAudioSummary,
    sttStatus,
    reasoningDebug,
  }: {
    harnessStatus: string;
    websocketDebug: string;
    websocketEvents: string[];
    liveTranscript: string;
    vadState: string;
    currentAudioLevel: number;
    audioDebugEvents: string[];
    transcriptEntries: TranscriptEntry[];
    lastDetection: string;
    harnessFrameCount: number;
    latestHarnessFrame: string | null;
    latestFaceCrop: string | null;
    latestFrameSummary: string;
    harnessAudioCount: number;
    latestAudioSummary: string;
    sttStatus: string;
    reasoningDebug: ChatDebug | null;
  } = $props();
</script>

<aside class="side-notes">
  <div class="note-card">
    <p class="card-label">Harness status</p>
    <p>{harnessStatus}</p>
  </div>
  <div class="note-card">
    <p class="card-label">Websocket</p>
    <p>{websocketDebug}</p>
    <div class="event-list">
      {#each websocketEvents as item}
        <span>{item}</span>
      {/each}
    </div>
  </div>
  <div class="note-card">
    <p class="card-label">Harness transcript</p>
    <p>{liveTranscript}</p>
  </div>
  <div class="note-card">
    <p class="card-label">Browser VAD</p>
    <p>{vadState}</p>
    <p>RMS {currentAudioLevel.toFixed(5)} / threshold {SPEECH_THRESHOLD}</p>
    <div class="event-list">
      {#each audioDebugEvents as item}
        <span>{item}</span>
      {/each}
    </div>
  </div>
  <div class="note-card">
    <TranscriptHistory entries={transcriptEntries} compact />
  </div>
  <div class="note-card">
    <p class="card-label">Face detection</p>
    <p>{lastDetection}</p>
  </div>
  <div class="note-card">
    <p class="card-label">Harness frame {harnessFrameCount}</p>
    {#if latestHarnessFrame}
      <img class="harness-frame" src={latestHarnessFrame} alt="Latest harness frame" />
    {/if}
    {#if latestFaceCrop}
      <img class="harness-frame face-crop" src={latestFaceCrop} alt="Latest detected face crop" />
    {/if}
    <p>{latestFrameSummary}</p>
  </div>
  <div class="note-card">
    <p class="card-label">Harness audio {harnessAudioCount}</p>
    <p>{latestAudioSummary}</p>
    <p>{sttStatus}</p>
  </div>
  {#if reasoningDebug}
    <div class="note-card reasoning-card">
      <p class="card-label">LLM reasoning</p>
      <p class="reasoning-meta">{reasoningDebug.provider ?? "no-provider"} / {reasoningDebug.model ?? "no-model"}</p>

      <p class="reasoning-heading">Current message</p>
      <pre>{reasoningDebug.current_message}</pre>

      <p class="reasoning-heading">Emotional context</p>
      <pre>{reasoningDebug.emotional_context}</pre>

      <p class="reasoning-heading">Transcript context</p>
      <pre>{reasoningDebug.transcript_lines.length ? reasoningDebug.transcript_lines.join("\n") : "[no transcript context]"}</pre>

      <p class="reasoning-heading">Prompt messages</p>
      <div class="event-list">
        {#each reasoningDebug.prompt_messages as prompt}
          <span><strong>{prompt.role}</strong>: {prompt.content}</span>
        {/each}
      </div>
    </div>
  {/if}
</aside>

<style>
  .side-notes {
    position: fixed;
    left: 1rem;
    bottom: 1rem;
    width: min(260px, calc(100vw - 2rem));
    max-height: calc(100vh - 2rem);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    z-index: 4;
  }

  .note-card {
    padding: 0.8rem 0.95rem;
    background: rgba(6, 10, 20, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    backdrop-filter: blur(12px);
  }

  .note-card p:last-child {
    color: rgba(255, 255, 255, 0.8);
    line-height: 1.45;
    margin-top: 0.3rem;
  }

  .card-label {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: rgba(255, 255, 255, 0.62);
  }

  .event-list {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    margin-top: 0.5rem;
    color: rgba(255, 255, 255, 0.62);
    font-size: 0.78rem;
    line-height: 1.3;
  }

  .harness-frame {
    width: 100%;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 0.65rem;
  }

  .face-crop {
    border-color: rgba(74, 222, 128, 0.45);
  }

  .reasoning-card pre {
    white-space: pre-wrap;
    word-break: break-word;
    margin-top: 0.35rem;
    color: rgba(255, 255, 255, 0.84);
    font-size: 0.8rem;
    line-height: 1.45;
  }

  .reasoning-meta,
  .reasoning-heading {
    font-size: 0.78rem;
    color: rgba(255, 255, 255, 0.62);
    margin-top: 0.45rem;
  }

  @media (max-width: 980px) {
    .side-notes {
      position: static;
      width: auto;
      padding: 0 1rem 1rem;
    }
  }
</style>
