<script lang="ts">
  import { browser } from "$app/environment";
  import { env as publicEnv } from "$env/dynamic/public";
  import { PUBLIC_HARNESS_WS_URL } from "$env/static/public";
  import { api, type ChatDebug, type ChatView, type Message, type Profile } from "$lib/api";
  // conversationState is still imported because it carries unrelated fields
  // (qaTurnCount, feedbackEvents, etc.) some other modules read. The
  // top-level `mode` / `stage` are no longer used here — the conductor
  // drives the surface via backendView. setMode / setStage retired.
  import { conversationState } from "$lib/conversation/store.svelte";
  import {
    checkInState,
    openCheckIn,
    closeCheckIn,
    advanceOverlayStep,
  } from "$lib/conversation/checkInState.svelte";
  import {
    SAMPLE_OVERLAY_CONVERSATIONAL,
    SAMPLE_OVERLAY_STATIC,
    SAMPLE_PAGE_SEQUENTIAL,
    SAMPLE_PAGE_ALL_AT_ONCE,
  } from "$lib/conversation/sampleCheckIns";
  import ChatInput from "$lib/components/ChatInput.svelte";
  import DebugDashboard from "$lib/components/DebugDashboard.svelte";
  import ModePanel from "$lib/components/ModePanel.svelte";
  import ProfileModal from "$lib/components/ProfileModal.svelte";
  import QuestionnairePage from "$lib/components/QuestionnairePage.svelte";
  import SpeakingCircle from "$lib/components/SpeakingCircle.svelte";
  import WebcamPreview from "$lib/components/WebcamPreview.svelte";
  import {
    deriveAssistantPhase,
    isRealTranscript,
    phaseLabel,
    shouldPromoteTranscript,
    transcriptPreview,
  } from "$lib/conversation/uiState";
  import { BrowserVadController } from "$lib/harness/browserVad";
  import {
    EMOTION_COLOURS,
    FRAME_INTERVAL_MS,
    SPEECH_THRESHOLD,
    formatTimings,
    isEmotion,
    type Emotion,
    type TranscriptEntry,
  } from "$lib/harness/types";

  const HARNESS_WS_URL = PUBLIC_HARNESS_WS_URL || "ws://127.0.0.1:8000/ws";

  // Mic-gating feature flag. When true (current behaviour): the moment the
  // user sends a chat message OR a form completes, the VAD pauses until
  // the assistant has finished thinking + speaking. When false, the user
  // can talk over a pending reply — useful for testing "barge-in" UX
  // later. Flip this to wire up that experiment without touching the
  // dozen call sites that drive assistantThinking.
  const LOCK_MIC_DURING_REPLY = true;
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
  let latestConfidence = $state<number | null>(null);
  type RejectedTranscript = {
    id: string;
    timestamp: string;
    text: string;
    reason: string;
    confidence: number | null;
    durationMs: number | null;
  };
  let rejectedTranscripts = $state<RejectedTranscript[]>([]);
  let audioDebugEvents = $state<string[]>([]);
  let currentAudioLevel = $state(0);
  let vadState = $state("Mic idle");
  const DEBUG_VISIBLE_KEY = "debug:visible";
  let showDebugDashboard = $state(DEBUG_ENV_ENABLED);

  function toggleDebugDashboard() {
    showDebugDashboard = !showDebugDashboard;
    if (browser) localStorage.setItem(DEBUG_VISIBLE_KEY, String(showDebugDashboard));
  }
  let pendingTranscript = $state<string | null>(null);
  let lastPromotedTranscript = $state<string | null>(null);
  let speechPulse = $state(0);
  let latestReasoningDebug = $state<ChatDebug | null>(null);
  // Backend-supplied view directive — the conductor's current decision about
  // what surface the frontend should render. Source of truth lives in the
  // model service; this is a read-only mirror.
  let backendView = $state<ChatView>({ surface: "chat" });

  // True while we're waiting for the assistant's yarn-opener reply after a
  // form completion. Drives the thinking indicator + mic pause. Cleared
  // when an assistant_reply (or assistant_reply_error) message arrives, or
  // when a safety timeout fires.
  let assistantThinking = $state(false);
  let assistantThinkingTimeout = $state<ReturnType<typeof setTimeout> | null>(null);

  function clearAssistantThinking() {
    assistantThinking = false;
    if (assistantThinkingTimeout) {
      clearTimeout(assistantThinkingTimeout);
      assistantThinkingTimeout = null;
    }
  }

  function startAssistantThinking() {
    assistantThinking = true;
    if (assistantThinkingTimeout) clearTimeout(assistantThinkingTimeout);
    // Safety net — claude-code subprocess may hang or never respond. After
    // 60s, give up and let the user speak again.
    assistantThinkingTimeout = setTimeout(() => {
      assistantThinking = false;
      assistantThinkingTimeout = null;
    }, 60_000);
  }

  let ws = $state<WebSocket | null>(null);
  let frameInterval = $state<ReturnType<typeof setInterval> | null>(null);
  let speechPulseTimeout = $state<ReturnType<typeof setTimeout> | null>(null);
  let browserVad: BrowserVadController | null = null;

  const bgColour = $derived(EMOTION_COLOURS[emotion]);
  const isOverlayActive = $derived(
    checkInState.active && checkInState.spec?.elevation === "overlay"
  );
  const isPageActive = $derived(
    checkInState.active && checkInState.spec?.elevation === "page"
  );

  function cancelMicIfRecording() {
    // Abort any in-progress utterance but keep the audio engine alive —
    // calling stop() here would close the AudioContext and never restart
    // (the auto-listen $effect bails because isListening is still true).
    browserVad?.cancelUtterance();
  }

  // Single funnel for every answer source — chip click, typed text, and
  // speech transcript all flow through here so multi-step overlays advance
  // consistently regardless of input mode.
  function handleCheckInAnswer(value: string) {
    void sendMessage(value);
    if (!checkInState.active) return;
    if (checkInState.spec?.elevation === "overlay") {
      // Multi-step: advance if there's a next step, otherwise close.
      // Single-step: advanceOverlayStep returns false → close.
      if (!advanceOverlayStep()) closeCheckIn();
    }
    // Page elevation keeps its own per-question state and doesn't close on
    // each answer; QuestionnairePage handles that.
  }

  // Page-elevation per-question hook. For now identical to the overlay path
  // (just send), but takes the questionId so once the reasoner is wired this
  // can carry per-question metadata.
  // Debug-fixture path (Shift+3/4 checkInState overlay). Ignores isLast — the
  // debug overlay is dismissed via Esc, not by form completion.
  function handlePageAnswer(_questionId: string, value: string, _isLast?: boolean) {
    void sendMessage(value);
  }

  // Send a typed system event over the harness WebSocket. The model service
  // appends it to the session's events buffer (merged into the LLM-facing
  // transcript stream) and, for kind="form_complete", steps the conductor
  // and pushes back a view_update.
  function sendSystemEvent(kind: string, payload: Record<string, unknown> = {}) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "system_event", kind, payload, t: Date.now() / 1000 }));
  }

  // Conductor-driven check-in path. Each chip click emits a form_answer
  // event; the final chip also emits form_complete so the conductor's
  // qa_form → next-state transition fires. No /chat round-trip for chip
  // clicks — the answers are structured signals, not user speech.
  // Text / STT during the form surface flows through QuestionnairePage:
  //   - chip match found → form_answer event + advance
  //   - matchless free-text → handleFormFreeText emits free_text_input
  // pendingFormInputText is set by STT (parent owns the transcript pipe);
  // composer-typed text reaches QuestionnairePage directly via ChatInput.
  let lastFreeTextNote = $state("");
  let pendingFormInputText = $state<string | null>(null);
  function handleFormFreeText(text: string) {
    const cleaned = text.trim();
    if (!cleaned) return;
    sendSystemEvent("free_text_input", { text: cleaned });
    lastFreeTextNote = cleaned;
  }

  function handleConductorPageAnswer(questionId: string, value: string, isLast: boolean) {
    sendSystemEvent("form_answer", { question_id: questionId, value });
    if (isLast) {
      // Empty payload — the backend conductor knows which form is current;
      // exposing the state name to the LLM via the rendered event would
      // violate "the LLM never sees state-machine vocabulary".
      sendSystemEvent("form_complete", {});
      // Note: deliberately no startAssistantThinking() here. After the
      // form, we want the user to be able to keep going immediately —
      // the yarn-opener LLM call runs in the background and the reply
      // will be spoken when it lands (which pauses the mic during TTS,
      // not during the wait).
    }
  }
  const assistantPhase = $derived(deriveAssistantPhase({
    backendOnline,
    profileReady: Boolean(profile),
    isListening,
    chatBusy,
    isSpeaking,
  }));
  const transcriptText = $derived(transcriptPreview(liveTranscript, transcriptEntries));
  // Indicator tracks MediaRecorder lifecycle: ON while the recorder is still
  // capturing (active speech OR the silence-wait grace period), OFF once
  // .stop() has been called or the clip has been sent.
  const isActivelyRecording = $derived(
    isListening && (
      vadState.startsWith("Recording speech")
      || vadState.startsWith("Silence detected")
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
      // User toggle preference wins over env/URL once they've expressed it.
      const stored = localStorage.getItem(DEBUG_VISIBLE_KEY);
      if (stored !== null) showDebugDashboard = stored === "true";
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

  // Held outside speak() so the utterance survives until the speech
  // engine fires onend — Chrome occasionally GCs in-flight utterances.
  let activeChatUtterance: SpeechSynthesisUtterance | null = null;
  function speak(text: string) {
    if (!browser || typeof window === "undefined" || !("speechSynthesis" in window)) return;
    const synth = window.speechSynthesis;
    // Chrome / Safari sometimes leave the synth in a paused state when
    // the tab regains focus — without resume() speak() is a no-op.
    if (synth.paused) synth.resume();
    const utterance = new SpeechSynthesisUtterance(text);
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
    utterance.onerror = (event) => {
      console.warn("[tts] chat reply failed:", event.error, text);
      isSpeaking = false;
      speechPulse = 0;
    };
    activeChatUtterance = utterance;
    // No cancel() — calling cancel followed by speak races in Chrome and
    // silently drops the utterance. If a previous reply is still
    // speaking, the new one queues; that's an acceptable trade.
    synth.speak(utterance);
  }

  async function sendMessage(text: string) {
    if (chatBusy) return;
    chatBusy = true;
    // Lock the mic the moment the message goes to the backend — gated by
    // LOCK_MIC_DURING_REPLY so the barge-in experiment can flip it later.
    if (LOCK_MIC_DURING_REPLY) startAssistantThinking();

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
    };
    messages = [...messages, userMsg];

    try {
      const { response, view, debug } = await api.sendChat(text);
      backendOnline = true;
      latestReasoningDebug = debug;
      backendView = view;
      const agentMsg: Message = {
        id: crypto.randomUUID(),
        role: "agent",
        content: response,
        timestamp: new Date().toISOString(),
      };
      messages = [...messages, agentMsg];
      // Clear the thinking flag — isSpeaking (from speak()) now drives
      // the mic gate while TTS reads the reply aloud.
      clearAssistantThinking();
      // If the conductor just transitioned us into a form, the form's
      // own TTS will read the first question — speaking the LLM's
      // wind-down reply on top of it overlaps. Skip TTS unless we're
      // staying on the chat surface.
      if (view.surface === "chat") {
        speak(response);
      }
    } catch {
      backendOnline = false;
      const errMsg: Message = {
        id: crypto.randomUUID(),
        role: "agent",
        content: "(Could not reach backend - is it running?)",
        timestamp: new Date().toISOString(),
      };
      messages = [...messages, errMsg];
      clearAssistantThinking();
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
        // Only display real transcripts. Filtered/placeholder strings like
        // [whisper.cpp transcript filtered] or [blank_audio] are dropped here
        // so the prior real transcript stays visible until the user produces
        // something new. shouldPromoteTranscript already guards chat sends.
        const text = msg.text as string | undefined;
        if (text && isRealTranscript(text)) {
          liveTranscript = text;
          if (shouldPromoteTranscript(text, lastPromotedTranscript)) {
            pendingTranscript = text;
            lastPromotedTranscript = text;
          }
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
      } else if (msg.type === "view_update") {
        // Conductor stepped on the model service (e.g. form_complete event)
        // and pushed us a new view. Mirror it; render reactively.
        if (msg.view) backendView = msg.view as ChatView;
      } else if (msg.type === "assistant_reply") {
        // Backend's yarn opener — assistant speaks first after a form
        // completion. Append to chat history + persist server-side so
        // subsequent /chat calls include it in the LLM's context.
        const text = String(msg.text ?? "").trim();
        if (text) {
          const agentMsg: Message = {
            id: crypto.randomUUID(),
            role: "agent",
            content: text,
            timestamp: new Date().toISOString(),
          };
          messages = [...messages, agentMsg];
          speak(text);
          void api.appendHistory(agentMsg).catch(() => { /* non-fatal */ });
        }
        clearAssistantThinking();
      } else if (msg.type === "assistant_reply_error") {
        // Backend's yarn opener failed. Surface a soft fallback so the
        // user isn't left waiting and can speak again.
        clearAssistantThinking();
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
        const rawConf = typeof msg.stt_confidence === "number" ? msg.stt_confidence : null;
        latestConfidence = rawConf;
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
        // Rejected-transcript log: surfaces what whisper actually heard for
        // clips that didn't pass the filter, so we can debug threshold tuning.
        if (msg.accepted === false && typeof msg.raw_text === "string" && msg.raw_text.trim() !== "") {
          rejectedTranscripts = [{
            id: crypto.randomUUID(),
            timestamp: new Date().toLocaleTimeString(),
            text: msg.raw_text,
            reason: (msg.stt_filter_reason as string | undefined) ?? "unknown",
            confidence: typeof msg.stt_confidence === "number" ? msg.stt_confidence : null,
            durationMs: typeof msg.duration_ms === "number" ? msg.duration_ms : null,
          }, ...rejectedTranscripts].slice(0, 10);
        }
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

    socket.onerror = (error) => {
      harnessOnline = false;
      lastDetection = "Harness unavailable";
      console.log(error)
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
      // Route speech depending on what surface is mounted:
      //   - debug Shift+overlay active → funnel into its step advancement
      //   - conductor-driven form surface → hand the text to
      //     QuestionnairePage via pendingFormInputText; it tries to
      //     match a chip on the current question, falls back to
      //     free-text if there's no match.
      //   - otherwise (yarn / chat) → normal chat turn
      if (isOverlayActive) {
        handleCheckInAnswer(nextTranscript);
      } else if (backendView.surface === "checkin") {
        pendingFormInputText = nextTranscript;
      } else {
        void sendMessage(nextTranscript);
      }
    }
  });

  $effect(() => {
    if (!profile || !harnessOnline || !micStream || isListening) return;
    isListening = true;
    startAudioStreaming();
  });

  // Pause the mic while the assistant is generating its yarn-opener OR
  // while TTS is speaking the reply aloud — we don't want the mic
  // picking up the speaker's own voice. The VAD still reports the audio
  // level while paused so the UI can show "I hear you but can't respond
  // yet" via the micGated/userTryingToSpeak flags below.
  const micGated = $derived(assistantThinking || isSpeaking);
  const userTryingToSpeak = $derived(micGated && currentAudioLevel > SPEECH_THRESHOLD);
  $effect(() => {
    if (micGated) browserVad?.pause();
    else browserVad?.resume();
  });

  // React to surface transitions driven by the conductor. The form's
  // speakPrompt manages its own TTS lifecycle (and sendMessage already
  // skips chat-reply TTS when view.surface !== "chat") — so we don't
  // need to cancel speechSynthesis here. We DO want to show the
  // "forming a response" lock when the user finishes a form and the
  // surface flips to chat while the yarn-opener LLM call runs on the
  // server. It clears when assistant_reply (or its error twin) arrives.
  let previousSurface: ChatView["surface"] = "chat";
  $effect(() => {
    const surface = backendView.surface;
    if (surface !== previousSurface) {
      const prev = previousSurface;
      previousSurface = surface;
      if (prev === "checkin" && surface === "chat") {
        startAssistantThinking();
      }
    }
  });

  $effect(() => {
    init();
  });

  // Debug-only keypresses to open sample check-ins without backend involvement.
  // Shift+1/2 → overlay variants. Shift+3/4 → full-page variants. Esc closes.
  // Gated on the env flag (not the dashboard's visible state) so the shortcuts
  // keep working when the dashboard panel is collapsed.
  $effect(() => {
    if (!DEBUG_ENV_ENABLED || !browser) return;

    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;

      if (e.shiftKey && e.code === "Digit1") {
        openCheckIn(SAMPLE_OVERLAY_CONVERSATIONAL, "debug");
      } else if (e.shiftKey && e.code === "Digit2") {
        openCheckIn(SAMPLE_OVERLAY_STATIC, "debug");
      } else if (e.shiftKey && e.code === "Digit3") {
        openCheckIn(SAMPLE_PAGE_SEQUENTIAL, "debug");
      } else if (e.shiftKey && e.code === "Digit4") {
        openCheckIn(SAMPLE_PAGE_ALL_AT_ONCE, "debug");
      } else if (e.key === "Escape" && checkInState.active) {
        closeCheckIn();
      }
    }

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
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

  {#if isPageActive && checkInState.spec?.elevation === "page"}
    <main class="page-mount">
      <QuestionnairePage
        spec={checkInState.spec}
        onAnswer={handlePageAnswer}
        onTextSubmit={sendMessage}
        onCancelMic={cancelMicIfRecording}
        {isListening}
        onMicToggle={toggleMic}
      />
    </main>
  {:else}
  <main class="main-layout" class:hero-recede={isOverlayActive}>
    {#if backendView.surface === "chat"}
      <section class="hero-shell">
        <div class="hero-copy">
          <p class="hero-kicker">{phaseLabel(assistantPhase)}</p>
          <h2>A calmer, voice-first conversation.</h2>
          <p class="helper-text">{bootMessage}</p>
          <div
            class="recording-subtitle"
            class:listening={isListening && !micGated}
            class:recording={isActivelyRecording}
            class:locked={micGated}
            class:locked-attempt={userTryingToSpeak}
            style={`--mic-pulse:${micPulse};`}
          >
            <span class="recording-dot" aria-hidden="true"></span>
            <span class="recording-copy">
              {#if userTryingToSpeak}
                I can hear you, but I can't reply until I finish — one moment.
              {:else if micGated}
                Mic is paused while I'm replying.
              {:else if isActivelyRecording}
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

        <SpeakingCircle phase={assistantPhase} pulse={speechPulse} compact={isOverlayActive} />

        <div class="transcript-shell">
          <p class="transcript-label">
            Live transcript
            {#if latestConfidence !== null}
              <span class="transcript-confidence">
                · STT confidence {(latestConfidence * 100).toFixed(0)}%
              </span>
            {/if}
          </p>
          <p class="transcript-text">{transcriptText}</p>
        </div>

        <div class="response-shell">
          <p class="response-label">Latest response</p>
          {#if assistantThinking}
            <p class="response-text thinking" aria-live="polite">
              <span class="thinking-dot"></span>
              <span class="thinking-dot"></span>
              <span class="thinking-dot"></span>
              <span class="thinking-label">Forming a response…</span>
            </p>
          {:else}
            <p class="response-text">{latestAssistantMessage}</p>
          {/if}
        </div>

        <div class="composer-shell">
          <ChatInput
            onSend={sendMessage}
            {isListening}
            onMicToggle={toggleMic}
            disabled={chatBusy || showModal || assistantThinking}
            audioLevel={currentAudioLevel}
            locked={micGated}
          />
          <p class="composer-hint">
            Speak to send a voice prompt automatically, or type if you want a quieter fallback.
          </p>
        </div>
      </section>
    {:else if backendView.surface === "checkin" && backendView.spec}
      <section class="checkin-mount">
        <QuestionnairePage
          spec={backendView.spec}
          onAnswer={handleConductorPageAnswer}
          onTextSubmit={handleFormFreeText}
          onCancelMic={cancelMicIfRecording}
          {isListening}
          onMicToggle={toggleMic}
          freeTextNote={lastFreeTextNote}
          audioLevel={currentAudioLevel}
          locked={micGated}
          pendingInputText={pendingFormInputText}
          onInputConsumed={() => (pendingFormInputText = null)}
        />
      </section>
    {:else if backendView.surface === "done"}
      <section class="mode-stub">
        <p class="hero-kicker">All done</p>
        <h2>Thanks for talking with us.</h2>
        <p class="helper-text">You can close this window now.</p>
      </section>
    {/if}
  </main>

  {#if isOverlayActive && checkInState.spec?.elevation === "overlay"}
    <div class="overlay-mount">
      <ModePanel
        spec={checkInState.spec}
        currentStep={checkInState.currentStep}
        onAnswer={(_stepId, value) => handleCheckInAnswer(value)}
        onCancelMic={cancelMicIfRecording}
        {isListening}
        onMicToggle={toggleMic}
      />
    </div>
  {/if}
  {/if}

  {#if DEBUG_ENV_ENABLED}
    <button class="debug-toggle" type="button" onclick={toggleDebugDashboard}>
      {showDebugDashboard ? "Hide debug" : "Show debug"}
    </button>
  {/if}

  {#if showDebugDashboard}
    <DebugDashboard
      {profile}
      {faceDetected}
      {lastDetection}
      {latestHarnessFrame}
      {latestFaceCrop}
      {latestFrameSummary}
      {emotion}
      {rejectedTranscripts}
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
    justify-content: flex-start;
    gap: 0.7rem;
    align-self: center;
    margin-top: 0.35rem;
    padding: 0.55rem 0.85rem;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.04);
    color: rgba(255, 255, 255, 0.68);
    /* Hold a stable width so the bars + dot don't slide as the copy
       length flips between "Listening for a strong enough signal…" and
       "I can hear you, but I can't reply until I finish — one moment." */
    min-width: min(440px, 90vw);
    transition:
      background 140ms ease,
      border-color 140ms ease,
      color 140ms ease,
      transform 140ms ease;
  }
  .recording-subtitle .recording-copy {
    flex: 1 1 auto;
  }
  .recording-subtitle .recording-bars {
    margin-left: auto;
    flex: 0 0 auto;
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

  /* Mic is paused while assistant is thinking or speaking. Purple to make
     "you can't talk right now" obviously distinct from listening / recording. */
  .recording-subtitle.locked {
    border-color: rgba(168, 85, 247, 0.28);
    background: rgba(124, 58, 237, 0.12);
    color: rgba(233, 213, 255, 0.94);
  }
  .recording-subtitle.locked .recording-dot {
    background: rgb(168, 85, 247);
  }
  .recording-subtitle.locked .recording-bars span {
    background: rgba(216, 180, 254, 0.55);
  }
  /* User is actively trying to speak through the lock — pulse to acknowledge
     we hear them even though we can't act on it. */
  .recording-subtitle.locked-attempt {
    border-color: rgba(192, 132, 252, 0.65);
    background: rgba(124, 58, 237, 0.22);
  }
  .recording-subtitle.locked-attempt .recording-dot {
    background: rgb(216, 180, 254);
    animation: record-pulse 1s ease-out infinite;
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

  .transcript-confidence {
    color: rgba(255, 255, 255, 0.5);
    font-weight: 400;
    text-transform: none;
    letter-spacing: normal;
  }

  .transcript-text {
    min-height: 1.6em;
  }

  .response-text {
    max-height: 4.8em;
    overflow: hidden;
  }

  .response-text.thinking {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    color: rgba(255, 255, 255, 0.85);
  }
  .thinking-dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.85);
    animation: thinkingPulse 1.2s ease-in-out infinite;
  }
  .thinking-dot:nth-child(2) { animation-delay: 0.18s; }
  .thinking-dot:nth-child(3) { animation-delay: 0.36s; }
  .thinking-label {
    margin-left: 0.4rem;
    font-style: italic;
    opacity: 0.75;
  }
  @keyframes thinkingPulse {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
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

  .debug-toggle {
    position: fixed;
    top: 1rem;
    left: 1rem;
    z-index: 60;
    background: rgba(15, 18, 28, 0.78);
    border: 1px solid rgba(255, 255, 255, 0.14);
    color: rgba(255, 255, 255, 0.82);
    border-radius: 999px;
    padding: 0.4rem 0.85rem;
    font-size: 0.78rem;
    cursor: pointer;
    backdrop-filter: blur(20px);
    transition: background 120ms ease, border-color 120ms ease;
  }
  .debug-toggle:hover {
    background: rgba(15, 18, 28, 0.92);
    border-color: rgba(255, 255, 255, 0.28);
  }

  /* Check-in overlay (elevation 1): floats over the hero, which recedes. */
  .main-layout.hero-recede .hero-shell {
    filter: brightness(0.78) saturate(0.85);
    transform: scale(0.97);
    transition: filter 280ms cubic-bezier(0.22, 1, 0.36, 1),
                transform 280ms cubic-bezier(0.22, 1, 0.36, 1);
  }
  .main-layout.hero-recede .transcript-shell,
  .main-layout.hero-recede .response-shell,
  .main-layout.hero-recede .composer-shell {
    opacity: 0;
    pointer-events: none;
    transition: opacity 220ms cubic-bezier(0.22, 1, 0.36, 1);
  }
  .main-layout .hero-shell {
    transition: filter 280ms cubic-bezier(0.22, 1, 0.36, 1),
                transform 280ms cubic-bezier(0.22, 1, 0.36, 1);
  }

  .overlay-mount {
    position: fixed;
    inset: 0;
    display: grid;
    place-items: center;
    padding: 1.5rem;
    z-index: 50;
    pointer-events: none;
  }
  .overlay-mount > :global(*) {
    pointer-events: auto;
  }

  /* Full-page check-in (elevation 2): replaces the conversation surface. */
  .page-mount {
    flex: 1;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: 2rem 1.5rem;
    overflow-y: auto;
    background:
      radial-gradient(circle at top, rgba(255, 255, 255, 0.06), transparent 40%);
  }

  @media (prefers-reduced-motion: reduce) {
    .main-layout .hero-shell,
    .main-layout.hero-recede .hero-shell,
    .main-layout.hero-recede .transcript-shell,
    .main-layout.hero-recede .response-shell,
    .main-layout.hero-recede .composer-shell {
      transition: none;
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
