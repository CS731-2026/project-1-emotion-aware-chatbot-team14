<script lang="ts">
  import type { Message, Profile } from "$lib/api";
  import type { TranscriptEntry } from "$lib/harness/types";
  import ChatHistory from "$lib/components/ChatHistory.svelte";
  import ChatInput from "$lib/components/ChatInput.svelte";
  import SpeakingCircle from "$lib/components/SpeakingCircle.svelte";
  import TranscriptHistory from "$lib/components/TranscriptHistory.svelte";
  import WebcamPreview from "$lib/components/WebcamPreview.svelte";

  let {
    profile,
    backendOnline,
    harnessOnline,
    faceDetected,
    bootMessage,
    harnessStatus,
    lastDetection,
    liveTranscript,
    currentAudioLevel,
    vadState,
    harnessAudioCount,
    latestAudioSummary,
    sttStatus,
    isSpeaking,
    messages,
    sendMessage,
    isListening,
    toggleMic,
    chatBusy,
    showModal,
    transcriptEntries,
    latestHarnessFrame,
    latestFaceCrop,
    latestFrameSummary,
    webcamStream,
  }: {
    profile: Profile | null;
    backendOnline: boolean;
    harnessOnline: boolean;
    faceDetected: boolean;
    bootMessage: string;
    harnessStatus: string;
    lastDetection: string;
    liveTranscript: string;
    currentAudioLevel: number;
    vadState: string;
    harnessAudioCount: number;
    latestAudioSummary: string;
    sttStatus: string;
    isSpeaking: boolean;
    messages: Message[];
    sendMessage: (text: string) => void;
    isListening: boolean;
    toggleMic: () => void;
    chatBusy: boolean;
    showModal: boolean;
    transcriptEntries: TranscriptEntry[];
    latestHarnessFrame: string | null;
    latestFaceCrop: string | null;
    latestFrameSummary: string;
    webcamStream: MediaStream | null;
  } = $props();
</script>

<div class="debug-shell">
  <div class="debug-topbar">
    <div>
      <p class="debug-label">Debug Dashboard</p>
      <h1>Connection and harness diagnostics</h1>
    </div>
    <div class="debug-profile">{profile?.name ?? "No profile selected"}</div>
  </div>

  <section class="debug-grid">
    <article class="debug-card" class:ok={backendOnline}>
      <span>Backend</span>
      <strong>{backendOnline ? "Online" : "Offline"}</strong>
      <p>{bootMessage}</p>
    </article>
    <article class="debug-card" class:ok={harnessOnline}>
      <span>Harness</span>
      <strong>{harnessOnline ? "Streaming" : "Offline"}</strong>
      <p>{harnessStatus}</p>
    </article>
    <article class="debug-card" class:ok={faceDetected}>
      <span>Face detection</span>
      <strong>{faceDetected ? "Detected" : "Waiting"}</strong>
      <p>{lastDetection}</p>
    </article>
    <article class="debug-card">
      <span>Transcript</span>
      <strong>Latest</strong>
      <p>{liveTranscript}</p>
    </article>
    <article class="debug-card">
      <span>Browser VAD</span>
      <strong>{currentAudioLevel.toFixed(5)}</strong>
      <p>{vadState}</p>
    </article>
    <article class="debug-card">
      <span>Audio debug</span>
      <strong>{harnessAudioCount} chunks</strong>
      <p>{latestAudioSummary}</p>
    </article>
    <article class="debug-card">
      <span>STT</span>
      <strong>whisper.cpp</strong>
      <p>{sttStatus}</p>
    </article>
  </section>

  <section class="debug-panels">
    <div class="debug-panel">
      <SpeakingCircle {isSpeaking} />
    </div>
    <div class="debug-panel debug-chat">
      <ChatHistory {messages} />
      <ChatInput onSend={sendMessage} {isListening} onMicToggle={toggleMic} disabled={chatBusy || showModal} />
    </div>
    <div class="debug-panel transcript-panel">
      <TranscriptHistory entries={transcriptEntries} />
    </div>
    <div class="debug-panel">
      {#if latestHarnessFrame}
        <img class="debug-frame" src={latestHarnessFrame} alt="Latest harness frame" />
        {#if latestFaceCrop}
          <img class="debug-frame face-crop" src={latestFaceCrop} alt="Latest detected face crop" />
        {/if}
        <p class="debug-frame-copy">{latestFrameSummary}</p>
      {:else}
        <WebcamPreview stream={webcamStream} />
        <p class="debug-frame-copy">{latestFrameSummary}</p>
      {/if}
    </div>
  </section>
</div>

<style>
  .debug-shell {
    min-height: 100vh;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .debug-label {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: rgba(255, 255, 255, 0.62);
  }

  .debug-topbar {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
  }

  .debug-topbar h1 {
    font-size: clamp(1.4rem, 4vw, 2rem);
    margin: 0.2rem 0;
  }

  .debug-profile,
  .debug-card,
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

  .debug-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 1rem;
  }

  .debug-card {
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .debug-card.ok {
    border-color: rgba(74, 222, 128, 0.35);
    background: rgba(34, 197, 94, 0.13);
  }

  .debug-card span {
    color: rgba(255, 255, 255, 0.65);
    font-size: 0.82rem;
  }

  .debug-card p {
    color: rgba(255, 255, 255, 0.82);
    line-height: 1.4;
  }

  .debug-panels {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: 0.8fr 1.4fr 0.8fr;
    gap: 1rem;
  }

  .debug-panel {
    padding: 1rem;
    min-height: 0;
  }

  .debug-chat {
    display: flex;
    flex-direction: column;
  }

  .debug-panel :global(.preview) {
    position: static;
    width: 100%;
    height: 100%;
    min-height: 240px;
  }

  .debug-frame {
    width: 100%;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 0.65rem;
  }

  .face-crop {
    border-color: rgba(74, 222, 128, 0.45);
  }

  .debug-frame-copy {
    color: rgba(255, 255, 255, 0.78);
    line-height: 1.4;
  }

  @media (max-width: 980px) {
    .debug-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .debug-panels {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 720px) {
    .debug-shell {
      padding: 1rem;
    }

    .debug-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
