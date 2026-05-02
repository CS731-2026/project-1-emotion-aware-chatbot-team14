<script lang="ts">
  import { browser } from "$app/environment";
  import { env as publicEnv } from "$env/dynamic/public";
  import { PUBLIC_HARNESS_WS_URL } from "$env/static/public";
  import { api, type Message, type Profile } from "$lib/api";
  import ChatHistory from "$lib/components/ChatHistory.svelte";
  import ChatInput from "$lib/components/ChatInput.svelte";
  import DebugDashboard from "$lib/components/DebugDashboard.svelte";
  import ProfileModal from "$lib/components/ProfileModal.svelte";
  import SideNotes from "$lib/components/SideNotes.svelte";
  import SpeakingCircle from "$lib/components/SpeakingCircle.svelte";
  import WebcamPreview from "$lib/components/WebcamPreview.svelte";
  import { BrowserVadController } from "$lib/harness/browserVad";
  import {
    EMOTION_COLOURS,
    EMOTION_MAP,
    FRAME_INTERVAL_MS,
    formatTimings,
    type Emotion,
    type TranscriptEntry,
  } from "$lib/harness/types";

  const HARNESS_WS_URL = PUBLIC_HARNESS_WS_URL || "ws://127.0.0.1:8000/ws";
  const DEBUG_ENV_ENABLED = publicEnv.PUBLIC_DEBUG_DASHBOARD === "true";

  let messages = $state<Message[]>([]);
  let emotion = $state<Emotion>("neutral");
  let isSpeaking = $state(false);
  let isListening = $state(false);
  let webcamStream = $state<MediaStream | null>(null);
  let micStream = $state<MediaStream | null>(null);
  let profile = $state<Profile | null>(null);
  let showModal = $state(false);
  let chatBusy = $state(false);
  let bootMessage = $state("Starting up...");
  let backendOnline = $state(false);
  let harnessOnline = $state(false);
  let faceDetected = $state(false);
  let lastDetection = $state("No frames processed yet");
  let liveTranscript = $state("Listening for harness transcript...");
  let latestHarnessFrame = $state<string | null>(null);
  let latestFaceCrop = $state<string | null>(null);
  let latestFrameSummary = $state("No harness frame returned yet");
  let latestAudioSummary = $state("No audio chunk returned yet");
  let harnessStatus = $state("Harness status not received yet");
  let harnessFrameCount = $state(0);
  let harnessAudioCount = $state(0);
  let sttStatus = $state("STT status unknown");
  let websocketDebug = $state("No websocket events received yet");
  let websocketEvents = $state<string[]>([]);
  let transcriptEntries = $state<TranscriptEntry[]>([]);
  let audioDebugEvents = $state<string[]>([]);
  let currentAudioLevel = $state(0);
  let vadState = $state("Mic idle");
  let showDebugDashboard = $state(DEBUG_ENV_ENABLED);

  let ws = $state<WebSocket | null>(null);
  let frameInterval = $state<ReturnType<typeof setInterval> | null>(null);
  let browserVad: BrowserVadController | null = null;

  const bgColour = $derived(EMOTION_COLOURS[emotion]);
  const statusText = $derived.by(() => {
    if (!backendOnline) return "Backend offline";
    if (!profile) return "Select a profile to begin";
    if (!harnessOnline) return "Backend ready, harness not connected";
    if (!faceDetected) return "Harness connected, waiting for face";
    return "Frontend, backend, and harness connected";
  });

  async function init() {
    browserVad = new BrowserVadController({
      getMicStream: () => micStream,
      getSocket: () => ws,
      onAudioLevel: (level) => (currentAudioLevel = level),
      onVadState: (state) => (vadState = state),
      onTranscriptStatus: (status) => (liveTranscript = status),
      onDebug: addAudioDebug,
    });

    if (browser) {
      const params = new URLSearchParams(window.location.search);
      if (params.get("debug") === "1") showDebugDashboard = true;
    }

    try {
      await api.getHealth();
      backendOnline = true;
      bootMessage = "Backend reachable";
    } catch {
      backendOnline = false;
      bootMessage = "Frontend is running without backend";
    }

    if (backendOnline) {
      try {
        const sess = await api.getSession();
        if (!sess.profileId) {
          showModal = true;
          bootMessage = "Backend reachable - select a profile to begin";
        } else {
          const [hist, profs] = await Promise.all([api.getHistory(), api.getProfiles()]);
          messages = hist;
          profile = profs.find((p) => p.id === sess.profileId) ?? null;
          if (sess.profileId) connectHarness(sess.profileId);
          bootMessage = "Frontend, backend, and session are ready";
        }
      } catch {
        showModal = true;
        bootMessage = "Backend responded, but session bootstrap failed";
      }
    } else {
      messages = [{
        id: crypto.randomUUID(),
        role: "agent",
        content: "Frontend booted successfully. Start the backend to enable profiles and chat.",
        timestamp: new Date().toISOString(),
      }];
    }

    initWebcam();
    initMic();
  }

  async function initWebcam() {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
    } catch {
      webcamStream = null;
    }
  }

  async function initMic() {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      micStream = null;
    }
  }

  function speak(text: string) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onstart = () => (isSpeaking = true);
    utterance.onend = () => (isSpeaking = false);
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  }

  async function sendMessage(text: string) {
    if (chatBusy) return;
    chatBusy = true;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
    };
    messages = [...messages, userMsg];

    try {
      const { response } = await api.sendChat(text);
      backendOnline = true;
      const agentMsg: Message = {
        id: crypto.randomUUID(),
        role: "agent",
        content: response,
        timestamp: new Date().toISOString(),
      };
      messages = [...messages, agentMsg];
      speak(response);
    } catch {
      backendOnline = false;
      const errMsg: Message = {
        id: crypto.randomUUID(),
        role: "agent",
        content: "(Could not reach backend - is it running?)",
        timestamp: new Date().toISOString(),
      };
      messages = [...messages, errMsg];
    } finally {
      chatBusy = false;
    }
  }

  async function onProfileSelected(p: Profile) {
    profile = p;
    showModal = false;
    const hist = await api.getHistory();
    messages = hist;
    connectHarness(p.id);
  }

  function toggleMic() {
    isListening = !isListening;
    if (isListening) {
      startAudioStreaming();
    } else {
      stopAudioStreaming();
    }
  }

  function connectHarness(profileId: string) {
    if (ws) ws.close();
    const socket = new WebSocket(HARNESS_WS_URL);
    ws = socket;

    socket.onopen = () => {
      harnessOnline = true;
      socket.send(JSON.stringify({ type: "session_start", profile_id: profileId }));
      startFrameStreaming(socket);
      if (isListening) startAudioStreaming();
    };

    socket.onmessage = (ev) => {
      const msg = JSON.parse(ev.data as string);
      const eventLine = `${new Date().toLocaleTimeString()} ${msg.type}`;
      websocketEvents = [eventLine, ...websocketEvents].slice(0, 8);
      websocketDebug = eventLine;

      if (msg.type === "connection_ack") {
        harnessStatus = msg.message;
      } else if (msg.type === "message_ack") {
        websocketDebug = `Harness received ${msg.message_type}`;
      } else if (msg.type === "error") {
        harnessStatus = msg.message;
        latestFrameSummary = msg.message;
        latestAudioSummary = msg.message;
      } else if (msg.type === "audio_received") {
        harnessAudioCount = Number(msg.audio_chunk_count ?? harnessAudioCount);
        latestAudioSummary =
          `Harness received audio chunk ${msg.audio_chunk_count} - ` +
          `${msg.byte_length} base64 chars`;
      } else if (msg.type === "emotion_update") {
        const mapped = EMOTION_MAP[msg.emotion as string];
        if (mapped) emotion = mapped;
      } else if (msg.type === "face_detection") {
        faceDetected = Boolean(msg.detected);
        lastDetection = msg.detected
          ? `Face detected at ${new Date().toLocaleTimeString()}`
          : `No face detected at ${new Date().toLocaleTimeString()}`;
      } else if (msg.type === "transcript_chunk") {
        liveTranscript = msg.text || "Harness received audio but produced no transcript";
      } else if (msg.type === "harness_status") {
        harnessStatus =
          `YOLO=${msg.face_detector_loaded ? "loaded" : "missing"} | ` +
          `device=${msg.face_detector_device ?? "unknown"} | ` +
          `${msg.face_detector_device_reason ?? "device status unknown"} | ` +
          `STT=${msg.stt_loaded ? "loaded" : "missing"} (${msg.stt_engine}/${msg.stt_model}) | ` +
          `test emotions=${msg.test_emotions ? "on" : "off"}`;
        sttStatus = msg.stt_loaded
          ? `whisper.cpp ready: ${msg.stt_engine}/${msg.stt_model}`
          : "whisper.cpp not loaded";
      } else if (msg.type === "frame_debug") {
        harnessFrameCount = Number(msg.frame_count ?? harnessFrameCount);
        latestHarnessFrame = `data:image/jpeg;base64,${msg.image_data}`;
        latestFaceCrop = msg.face_crop_data ? `data:image/jpeg;base64,${msg.face_crop_data}` : null;
        latestFrameSummary =
          `Harness frame ${msg.frame_count} - ` +
          `${msg.detected ? "face detected" : "no face detected"} - ` +
          `${msg.detector_loaded ? "YOLO loaded" : "YOLO missing"}` +
          `${msg.detector_device ? ` on ${msg.detector_device}` : ""}` +
          `${msg.box ? ` - box ${JSON.stringify(msg.box)}` : ""}` +
          `${msg.timings_ms ? ` - timings ${formatTimings(msg.timings_ms)}` : ""}`;
      } else if (msg.type === "audio_debug") {
        harnessAudioCount = Number(msg.audio_chunk_count ?? harnessAudioCount);
        const timings = formatTimings(msg.timings_ms);
        sttStatus = msg.stt_error
          ? `whisper.cpp error: ${msg.stt_error}`
          : `${msg.stt_loaded ? "whisper.cpp ran" : "whisper.cpp missing"} (${msg.stt_engine})`;
        latestAudioSummary =
          `Harness audio chunk ${msg.audio_chunk_count} - ` +
          `${msg.byte_length} base64 chars - ${msg.text}` +
          `${timings ? ` - timings ${timings}` : ""}`;
        audioDebugEvents = [
          `${new Date().toLocaleTimeString()} chunk ${msg.audio_chunk_count}: ${msg.stt_error ? "error" : "done"} ${timings}`,
          ...audioDebugEvents,
        ].slice(0, 8);
        transcriptEntries = [{
          id: crypto.randomUUID(),
          text: msg.text || "[no transcript text]",
          timestamp: new Date().toLocaleTimeString(),
          chunk: Number(msg.audio_chunk_count ?? harnessAudioCount) || null,
          error: msg.stt_error ?? null,
          timings,
        }, ...transcriptEntries].slice(0, 12);
      }
    };

    socket.onclose = () => {
      ws = null;
      harnessOnline = false;
      faceDetected = false;
      stopAudioStreaming();
      if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
      }
    };

    socket.onerror = () => {
      harnessOnline = false;
      lastDetection = "Harness unavailable";
    };
  }

  function startFrameStreaming(socket: WebSocket) {
    const canvas = document.createElement("canvas");

    frameInterval = setInterval(() => {
      if (socket.readyState !== WebSocket.OPEN) return;
      const videoEl = document.querySelector("video") as HTMLVideoElement | null;

      if (videoEl && webcamStream) {
        canvas.width = videoEl.videoWidth || 320;
        canvas.height = videoEl.videoHeight || 240;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(videoEl, 0, 0);
          const jpeg = canvas.toDataURL("image/jpeg", 0.6).split(",")[1];
          socket.send(JSON.stringify({
            type: "video_frame",
            data: jpeg,
            timestamp: Date.now() / 1000,
          }));
          if (lastDetection === "No frames processed yet") {
            lastDetection = "Frames are being sent to the harness";
          }
        }
      } else if (webcamStream && !videoEl) {
        lastDetection = "Waiting for webcam element to mount";
      } else if (!webcamStream) {
        lastDetection = "Waiting for webcam permission";
      }
    }, FRAME_INTERVAL_MS);
  }

  function addAudioDebug(message: string) {
    audioDebugEvents = [`${new Date().toLocaleTimeString()} ${message}`, ...audioDebugEvents].slice(0, 8);
  }

  function startAudioStreaming() {
    browserVad?.start();
  }

  function stopAudioStreaming() {
    browserVad?.stop(isListening);
  }

  $effect(() => {
    init();
  });

  $effect(() => {
    return () => {
      if (ws) {
        ws.close();
        ws = null;
      }
      browserVad?.destroy();
      if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
      }
    };
  });
</script>

<div class="app" style="background-color: {bgColour}">
  {#if showModal}
    <ProfileModal onSelected={onProfileSelected} />
  {/if}

  {#if showDebugDashboard}
    <DebugDashboard
      {profile}
      {backendOnline}
      {harnessOnline}
      {faceDetected}
      {bootMessage}
      {harnessStatus}
      {lastDetection}
      {liveTranscript}
      {currentAudioLevel}
      {vadState}
      {harnessAudioCount}
      {latestAudioSummary}
      {sttStatus}
      {isSpeaking}
      {messages}
      sendMessage={sendMessage}
      {isListening}
      toggleMic={toggleMic}
      {chatBusy}
      {showModal}
      {transcriptEntries}
      {latestHarnessFrame}
      {latestFaceCrop}
      {latestFrameSummary}
      {webcamStream}
    />
  {:else}
    <header class="topbar">
      <div class="title-block">
        <p class="eyebrow">Emotion-aware study companion</p>
        <h1>{profile?.name ?? "Study Companion"}</h1>
        <p class="status-line">{statusText}</p>
      </div>
      <div class="status-pills">
        <span class:ok={backendOnline} class="pill">Backend</span>
        <span class:ok={harnessOnline} class="pill">Harness</span>
        <span class:ok={faceDetected} class="pill">Face</span>
      </div>
    </header>

    <main class="main-layout">
      <section class="circle-zone">
        <SpeakingCircle {isSpeaking} />
        <p class="speech-state">{isSpeaking ? "Agent speaking" : "Agent idle"}</p>
        <p class="helper-text">{bootMessage}</p>
      </section>

      <section class="chat-shell">
        <ChatHistory {messages} />
        <ChatInput onSend={sendMessage} {isListening} onMicToggle={toggleMic} disabled={chatBusy || showModal} />
      </section>
    </main>

    <SideNotes
      {harnessStatus}
      {websocketDebug}
      {websocketEvents}
      {liveTranscript}
      {vadState}
      {currentAudioLevel}
      {audioDebugEvents}
      {transcriptEntries}
      {lastDetection}
      {harnessFrameCount}
      {latestHarnessFrame}
      {latestFaceCrop}
      {latestFrameSummary}
      {harnessAudioCount}
      {latestAudioSummary}
      {sttStatus}
    />

    <WebcamPreview stream={webcamStream} />
  {/if}
</div>

<style>
  .app {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    color: #f5f7fb;
    transition: background-color 1.5s ease;
    overflow: hidden;
  }

  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    padding: 1.25rem 1.25rem 0.75rem;
  }

  .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: rgba(255, 255, 255, 0.62);
  }

  .title-block h1 {
    font-size: clamp(1.4rem, 4vw, 2rem);
    margin: 0.2rem 0;
  }

  .status-line,
  .helper-text {
    color: rgba(255, 255, 255, 0.7);
  }

  .status-pills {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .pill {
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 999px;
    padding: 0.35rem 0.65rem;
    background: rgba(255, 255, 255, 0.06);
    font-size: 0.8rem;
  }

  .pill.ok {
    background: rgba(74, 222, 128, 0.18);
    border-color: rgba(74, 222, 128, 0.35);
  }

  .main-layout {
    flex: 1;
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    padding: 0 1rem 1rem;
    min-height: 0;
  }

  .circle-zone {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.25rem 0 0.75rem;
    text-align: center;
  }

  .speech-state {
    margin-top: 0.3rem;
    font-size: 0.95rem;
  }

  .chat-shell {
    width: min(860px, 100%);
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    min-height: 0;
    padding: 0.75rem 1rem 1rem;
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 24px;
    background: rgba(6, 10, 20, 0.26);
    backdrop-filter: blur(12px);
  }

  @media (max-width: 720px) {
    .topbar {
      flex-direction: column;
    }

    .main-layout {
      padding: 0 0.75rem 0.75rem;
    }

    .chat-shell {
      border-radius: 18px;
      padding: 0.5rem 0.75rem 0.75rem;
    }
  }
</style>
