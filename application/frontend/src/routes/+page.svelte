<script lang="ts">
  import { browser } from "$app/environment";
  import { env as publicEnv } from "$env/dynamic/public";
  import { PUBLIC_HARNESS_WS_URL } from "$env/static/public";
  import { api, type ChatDebug, type Message, type Profile } from "$lib/api";
  import { conversationState, setMode, setStage } from "$lib/conversation/store.svelte";
  import ChatInput from "$lib/components/ChatInput.svelte";
  import DebugDashboard from "$lib/components/DebugDashboard.svelte";
  import ProfileModal from "$lib/components/ProfileModal.svelte";
  import SpeakingCircle from "$lib/components/SpeakingCircle.svelte";
  import WebcamPreview from "$lib/components/WebcamPreview.svelte";
  import {
    deriveAssistantPhase,
    phaseLabel,
    shouldPromoteTranscript,
    transcriptPreview,
  } from "$lib/conversation/uiState";
  import { BrowserVadController } from "$lib/harness/browserVad";
  import {
    EMOTION_COLOURS,
    FRAME_INTERVAL_MS,
    formatTimings,
    isEmotion,
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
  let pendingTranscript = $state<string | null>(null);
  let lastPromotedTranscript = $state<string | null>(null);
  let speechPulse = $state(0);
  let latestReasoningDebug = $state<ChatDebug | null>(null);

  let ws = $state<WebSocket | null>(null);
  let frameInterval = $state<ReturnType<typeof setInterval> | null>(null);
  let speechPulseTimeout = $state<ReturnType<typeof setTimeout> | null>(null);
  let browserVad: BrowserVadController | null = null;

  const bgColour = $derived(EMOTION_COLOURS[emotion]);
  const assistantPhase = $derived(deriveAssistantPhase({
    backendOnline,
    profileReady: Boolean(profile),
    isListening,
    chatBusy,
    isSpeaking,
  }));
  const transcriptText = $derived(transcriptPreview(liveTranscript, transcriptEntries));
  const isActivelyRecording = $derived(
    isListening && (
      vadState.startsWith("Recording speech")
      || vadState.startsWith("Silence detected")
      || vadState.startsWith("Stopping speech clip")
      || vadState.startsWith("Clip sent")
    )
  );
  const micPulse = $derived(
    isListening ? Math.min(1, Math.max(0.08, currentAudioLevel / 0.012)) : 0
  );
  const latestAssistantMessage = $derived.by(() =>
    [...messages].reverse().find((message) => message.role === "agent")?.content
      ?? "I’ll respond out loud here once the backend reply comes through."
  );
  const statusText = $derived.by(() => {
    if (!backendOnline) return "Backend offline";
    if (!profile) return "Select a profile to begin";
    if (!harnessOnline) return "Backend ready, harness not connected";
    if (!micStream) return "Allow microphone access to start speaking";
    if (isListening) return "Listening for your voice";
    if (!faceDetected) return "Harness connected, waiting for face";
    if (chatBusy) return "Backend is forming a reply";
    return "Ready for a live conversation";
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
    if (!browser || typeof window === "undefined" || !("speechSynthesis" in window)) return;

    const utterance = new SpeechSynthesisUtterance(text);
    const preferredVoice = window.speechSynthesis.getVoices()
      .find((voice) => voice.lang.toLowerCase().startsWith("en"));

    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }

    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.volume = 1;

    utterance.onstart = () => {
      isSpeaking = true;
      speechPulse = 0.4;
    };
    utterance.onboundary = (event) => {
      if (event.name === "word" || event.charLength) {
        const emphasis = Math.min(1, Math.max(0.2, (event.charLength || 4) / 8));
        speechPulse = emphasis;
        if (speechPulseTimeout) clearTimeout(speechPulseTimeout);
        speechPulseTimeout = setTimeout(() => {
          speechPulse = 0.22;
        }, 120);
      }
    };
    utterance.onend = () => {
      isSpeaking = false;
      speechPulse = 0;
    };
    utterance.onerror = () => {
      isSpeaking = false;
      speechPulse = 0;
    };
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
      const { response, next_mode, next_stage, debug } = await api.sendChat(
        text,
        conversationState.mode,
        conversationState.stage,
      );
      backendOnline = true;
      latestReasoningDebug = debug;
      const agentMsg: Message = {
        id: crypto.randomUUID(),
        role: "agent",
        content: response,
        timestamp: new Date().toISOString(),
      };
      messages = [...messages, agentMsg];
      speak(response);
      // Reasoner decides where the conversation goes next. Apply transitions
      // to the store; the view block below re-renders to match.
      if (next_mode !== conversationState.mode) setMode(next_mode);
      if (next_stage !== conversationState.stage) setStage(next_stage);
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
    bootMessage = `Profile selected: ${p.name}`;
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
      bootMessage = "Harness connected";
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
        if (isEmotion(msg.emotion)) emotion = msg.emotion;
      } else if (msg.type === "face_detection") {
        faceDetected = Boolean(msg.detected);
        lastDetection = msg.detected
          ? `Face detected at ${new Date().toLocaleTimeString()}`
          : `No face detected at ${new Date().toLocaleTimeString()}`;
      } else if (msg.type === "transcript_chunk") {
        liveTranscript = msg.text || "Harness received audio but produced no transcript";
        if (msg.text && shouldPromoteTranscript(msg.text, lastPromotedTranscript)) {
          pendingTranscript = msg.text;
          lastPromotedTranscript = msg.text;
        }
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
    if (pendingTranscript && !chatBusy) {
      const nextTranscript = pendingTranscript;
      pendingTranscript = null;
      void sendMessage(nextTranscript);
    }
  });

  $effect(() => {
    if (!profile || !harnessOnline || !micStream || isListening) return;
    isListening = true;
    startAudioStreaming();
  });

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
      if (speechPulseTimeout) {
        clearTimeout(speechPulseTimeout);
        speechPulseTimeout = null;
      }
      if (browser && typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    };
  });
</script>

<div class="app" style="background-color: {bgColour}">
  {#if showModal}
    <ProfileModal onSelected={onProfileSelected} />
  {/if}

  <header class="topbar">
    <div class="title-block">
      <p class="eyebrow">Emotion-aware empathy bot</p>
      <h1>{profile?.name ?? "Empathy Bot"}</h1>
      <p class="status-line">{statusText}</p>
    </div>
    <div class="status-pills">
      <span class:ok={backendOnline} class="pill">Backend</span>
      <span class:ok={harnessOnline} class="pill">Harness</span>
      <span class:ok={faceDetected} class="pill">Face</span>
    </div>
  </header>

  <main class="main-layout">
    {#if conversationState.mode === "qa"}
      <section class="hero-shell">
        <div class="hero-copy">
          <p class="hero-kicker">{phaseLabel(assistantPhase)}</p>
          <h2>A calmer, voice-first conversation.</h2>
          <p class="helper-text">{bootMessage}</p>
          <div
            class="recording-subtitle"
            class:listening={isListening}
            class:recording={isActivelyRecording}
            style={`--mic-pulse:${micPulse};`}
          >
            <span class="recording-dot" aria-hidden="true"></span>
            <span class="recording-copy">
              {#if isActivelyRecording}
                Frontend is recording your voice
              {:else if isListening}
                Listening for a strong enough signal to record
              {:else}
                Voice capture is idle
              {/if}
            </span>
            <span class="recording-bars" aria-hidden="true">
              <span></span>
              <span></span>
              <span></span>
            </span>
          </div>
        </div>

        <SpeakingCircle phase={assistantPhase} pulse={speechPulse} />

        <div class="transcript-shell">
          <p class="transcript-label">Live transcript</p>
          <p class="transcript-text">{transcriptText}</p>
        </div>

        <div class="response-shell">
          <p class="response-label">Latest response</p>
          <p class="response-text">{latestAssistantMessage}</p>
        </div>

        <div class="composer-shell">
          <ChatInput onSend={sendMessage} {isListening} onMicToggle={toggleMic} disabled={chatBusy || showModal} />
          <p class="composer-hint">
            Speak to send a voice prompt automatically, or type if you want a quieter fallback.
          </p>
        </div>
      </section>
    {:else if conversationState.mode === "feedback"}
      <!-- Phase 5 will replace this stub with a timed self-report flow. -->
      <section class="mode-stub">
        <p class="hero-kicker">Feedback check-in</p>
        <h2>How are you feeling right now?</h2>
        <p class="helper-text">Self-report UI lands in Phase 5.</p>
      </section>
    {:else if conversationState.mode === "consent"}
      <!-- Phase 7 will replace this stub with consent + profile select. -->
      <section class="mode-stub">
        <p class="hero-kicker">Before we begin</p>
        <h2>Consent flow placeholder.</h2>
        <p class="helper-text">Consent + profile selection ships in Phase 7.</p>
      </section>
    {:else if conversationState.mode === "done"}
      <section class="mode-stub">
        <p class="hero-kicker">All done</p>
        <h2>Thanks for talking with us.</h2>
        <p class="helper-text">You can close this window now.</p>
      </section>
    {/if}
  </main>

  {#if showDebugDashboard}
    <DebugDashboard
      {profile}
      {faceDetected}
      {lastDetection}
      {latestHarnessFrame}
      {latestFaceCrop}
      {latestFrameSummary}
      {emotion}
      reasoningDebug={latestReasoningDebug}
    />
  {/if}

  <WebcamPreview stream={webcamStream} hidden />
</div>

<style>
  .app {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    color: #f5f7fb;
    background:
      radial-gradient(circle at top, rgba(255, 255, 255, 0.08), transparent 42%),
      linear-gradient(180deg, rgba(0, 0, 0, 0.05), rgba(0, 0, 0, 0.28));
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
    font-size: clamp(1.6rem, 4vw, 2.4rem);
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
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 1rem 1.5rem;
    min-height: 0;
  }

  .hero-shell {
    width: min(980px, 100%);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    padding: 1.5rem 1rem 1rem;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 32px;
    background: rgba(7, 10, 18, 0.24);
    backdrop-filter: blur(18px);
  }

  .mode-stub {
    width: min(640px, 100%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.6rem;
    padding: 2rem 1.5rem;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 28px;
    background: rgba(7, 10, 18, 0.24);
    backdrop-filter: blur(18px);
  }

  .mode-stub h2 {
    font-size: clamp(1.6rem, 4vw, 2.4rem);
    line-height: 1.1;
    margin: 0;
    letter-spacing: -0.02em;
  }

  .hero-copy {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    max-width: 42rem;
  }

  .hero-kicker,
  .transcript-label,
  .response-label {
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.75rem;
    color: rgba(255, 255, 255, 0.64);
  }

  .hero-copy h2 {
    font-size: clamp(2rem, 5vw, 3.8rem);
    line-height: 1;
    letter-spacing: -0.04em;
  }

  .recording-subtitle {
    --mic-pulse: 0;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.7rem;
    align-self: center;
    margin-top: 0.35rem;
    padding: 0.55rem 0.85rem;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.04);
    color: rgba(255, 255, 255, 0.68);
    transition:
      background 140ms ease,
      border-color 140ms ease,
      color 140ms ease,
      transform 140ms ease;
  }

  .recording-subtitle.listening {
    border-color: rgba(255, 255, 255, 0.14);
    background: rgba(255, 255, 255, 0.06);
    color: rgba(255, 255, 255, 0.8);
  }

  .recording-subtitle.recording {
    border-color: rgba(255, 132, 132, 0.28);
    background: rgba(255, 82, 82, 0.08);
    color: rgba(255, 255, 255, 0.94);
    transform: translateY(calc(var(--mic-pulse) * -1px));
  }

  .recording-dot {
    width: 0.62rem;
    height: 0.62rem;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.34);
    box-shadow: 0 0 0 rgba(255, 92, 92, 0);
    transition: background 140ms ease;
  }

  .recording-subtitle.listening .recording-dot {
    background: rgba(255, 255, 255, 0.76);
  }

  .recording-subtitle.recording .recording-dot {
    background: rgb(255, 108, 108);
    animation: record-pulse 1s ease-out infinite;
  }

  .recording-copy {
    font-size: 0.88rem;
    line-height: 1.3;
  }

  .recording-bars {
    display: inline-flex;
    align-items: flex-end;
    gap: 0.18rem;
    height: 0.8rem;
  }

  .recording-bars span {
    width: 0.16rem;
    height: calc(0.28rem + var(--mic-pulse) * 0.55rem);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.26);
    transform-origin: bottom;
    transition: height 120ms linear, background 140ms ease;
  }

  .recording-subtitle.listening .recording-bars span {
    background: rgba(255, 255, 255, 0.48);
  }

  .recording-subtitle.recording .recording-bars span:nth-child(1) {
    animation: level-bounce 0.85s ease-in-out infinite;
  }

  .recording-subtitle.recording .recording-bars span:nth-child(2) {
    animation: level-bounce 0.85s ease-in-out 0.12s infinite;
  }

  .recording-subtitle.recording .recording-bars span:nth-child(3) {
    animation: level-bounce 0.85s ease-in-out 0.24s infinite;
  }

  .transcript-shell,
  .response-shell {
    width: min(700px, 100%);
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 1rem 1.1rem;
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  .transcript-text,
  .response-text {
    font-size: 1rem;
    line-height: 1.6;
    color: rgba(255, 255, 255, 0.94);
  }

  .transcript-text {
    min-height: 1.6em;
  }

  .response-text {
    max-height: 4.8em;
    overflow: hidden;
  }

  .composer-shell {
    width: min(760px, 100%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
  }

  .composer-hint {
    color: rgba(255, 255, 255, 0.68);
    font-size: 0.88rem;
    line-height: 1.5;
    max-width: 38rem;
  }

  @keyframes record-pulse {
    0% {
      box-shadow: 0 0 0 0 rgba(255, 92, 92, 0.46);
    }
    100% {
      box-shadow: 0 0 0 10px rgba(255, 92, 92, 0);
    }
  }

  @keyframes level-bounce {
    0%, 100% {
      transform: scaleY(0.7);
    }
    50% {
      transform: scaleY(calc(1 + var(--mic-pulse) * 0.6));
    }
  }

  @media (max-width: 720px) {
    .topbar {
      flex-direction: column;
    }

    .main-layout {
      padding: 0 0.75rem 0.75rem;
    }

    .hero-shell {
      padding: 1.25rem 0.9rem 0.9rem;
      border-radius: 24px;
    }

    .hero-copy h2 {
      font-size: clamp(1.7rem, 9vw, 2.4rem);
    }

    .recording-subtitle {
      width: 100%;
      gap: 0.55rem;
      padding: 0.5rem 0.7rem;
    }

    .recording-copy {
      font-size: 0.82rem;
    }

    .transcript-shell,
    .response-shell {
      padding: 0.9rem 1rem;
    }
  }
</style>
