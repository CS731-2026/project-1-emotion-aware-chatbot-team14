<script lang="ts">
  import { api, type Message, type Profile } from "$lib/api";
  import ChatHistory from "$lib/components/ChatHistory.svelte";
  import ChatInput from "$lib/components/ChatInput.svelte";
  import SpeakingCircle from "$lib/components/SpeakingCircle.svelte";
  import WebcamPreview from "$lib/components/WebcamPreview.svelte";
  import ProfileModal from "$lib/components/ProfileModal.svelte";
  import { PUBLIC_HARNESS_WS_URL } from "$env/static/public";

  type Emotion = "neutral" | "happy" | "sad" | "angry" | "fearful" | "disgusted" | "surprised";

  const EMOTION_COLOURS: Record<Emotion, string> = {
    neutral:   "hsl(220, 15%, 10%)",
    happy:     "hsl(45, 70%, 15%)",
    sad:       "hsl(210, 60%, 12%)",
    angry:     "hsl(0, 65%, 15%)",
    fearful:   "hsl(275, 50%, 13%)",
    disgusted: "hsl(120, 35%, 11%)",
    surprised: "hsl(180, 55%, 13%)",
  };

  let messages       = $state<Message[]>([]);
  let emotion        = $state<Emotion>("neutral");
  let isSpeaking     = $state(false);
  let isListening    = $state(false);
  let webcamStream   = $state<MediaStream | null>(null);
  let micStream      = $state<MediaStream | null>(null);
  let profile        = $state<Profile | null>(null);
  let showModal      = $state(false);
  let chatBusy       = $state(false);

  // WebSocket to harness
  let ws            = $state<WebSocket | null>(null);
  let frameInterval = $state<ReturnType<typeof setInterval> | null>(null);
  const FRAME_INTERVAL_MS = 500; // throttled to 2fps

  const EMOTION_MAP: Record<string, Emotion> = {
    angry: "angry", disgust: "disgusted", fear: "fearful",
    happy: "happy", sad: "sad", surprise: "surprised", neutral: "neutral",
  };

  const bgColour = $derived(EMOTION_COLOURS[emotion]);

  async function init() {
    const sess = await api.getSession();
    if (!sess.profileId) {
      showModal = true;
    } else {
      const [hist, profs] = await Promise.all([api.getHistory(), api.getProfiles()]);
      messages = hist;
      profile = profs.find((p) => p.id === sess.profileId) ?? null;
      if (sess.profileId) connectHarness(sess.profileId);
    }
    initWebcam();
    initMic();
  }

  async function initWebcam() {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
    } catch {
      // Permission denied — webcamStream stays null
    }
  }

  async function initMic() {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      // Permission denied — micStream stays null
    }
  }

  function speak(text: string) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onstart = () => (isSpeaking = true);
    utterance.onend   = () => (isSpeaking = false);
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
      const agentMsg: Message = {
        id: crypto.randomUUID(),
        role: "agent",
        content: response,
        timestamp: new Date().toISOString(),
      };
      messages = [...messages, agentMsg];
      speak(response);
    } catch {
      const errMsg: Message = {
        id: crypto.randomUUID(),
        role: "agent",
        content: "(Could not reach backend — is it running?)",
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
  }

  function connectHarness(profileId: string) {
    if (ws) ws.close();
    const socket = new WebSocket(PUBLIC_HARNESS_WS_URL);
    ws = socket;

    socket.onopen = () => {
      socket.send(JSON.stringify({ type: "session_start", profile_id: profileId }));
      startFrameStreaming(socket);
    };

    socket.onmessage = (ev) => {
      const msg = JSON.parse(ev.data as string);
      if (msg.type === "emotion_update") {
        const mapped = EMOTION_MAP[msg.emotion as string];
        if (mapped) emotion = mapped;
      }
    };

    socket.onclose = () => {
      ws = null;
      if (frameInterval) { clearInterval(frameInterval); frameInterval = null; }
    };
  }

  function startFrameStreaming(socket: WebSocket) {
    const videoEl = document.querySelector("video") as HTMLVideoElement | null;
    const canvas = document.createElement("canvas");

    frameInterval = setInterval(() => {
      if (socket.readyState !== WebSocket.OPEN) return;

      // Send video frame
      if (videoEl && webcamStream) {
        canvas.width  = videoEl.videoWidth  || 320;
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
        }
      }
    }, FRAME_INTERVAL_MS);
  }

  $effect(() => { init(); });

  $effect(() => {
    return () => {
      // Cleanup on unmount
      if (ws) { ws.close(); ws = null; }
      if (frameInterval) { clearInterval(frameInterval); frameInterval = null; }
    };
  });
</script>

<div class="app" style="background-color: {bgColour}">
  {#if showModal}
    <ProfileModal onSelected={onProfileSelected} />
  {/if}

  <header class="topbar">
    <span class="title">Study Companion</span>
    {#if profile}
      <span class="profile-chip">{profile.name}</span>
    {/if}
  </header>

  <SpeakingCircle {isSpeaking} />

  <ChatHistory {messages} />

  <ChatInput onSend={sendMessage} {isListening} onMicToggle={toggleMic} disabled={chatBusy || showModal} />

  <WebcamPreview stream={webcamStream} />
</div>

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    transition: background-color 1.5s ease;
    overflow: hidden;
  }

  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1rem;
    border-bottom: 1px solid var(--color-border);
    background: rgba(0, 0, 0, 0.2);
    flex-shrink: 0;
  }

  .title {
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.02em;
  }

  .profile-chip {
    font-size: 0.8rem;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    color: var(--color-text-muted);
  }
</style>
