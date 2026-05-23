<!--
  /emotion-test/ — isolated emotion-bot harness.

  Reuses the model_service's WS endpoint and the production face
  detector → emotion classifier path, but skips chat, profiles, LLM,
  and audio/VAD entirely. Useful for testing emotion model changes
  without the rest of the app surface noise.

  No backend deps beyond the model_service running on port 8000.
-->
<script lang="ts">
  import { browser } from "$app/environment";
  import { PUBLIC_HARNESS_WS_URL } from "$env/static/public";
  import {
    EMOTION_COLOURS,
    EMOTION_EMOJI,
    EMOTION_LABEL,
    FRAME_INTERVAL_MS,
    isEmotion,
    type Emotion,
  } from "$lib/harness/types";

  const WS_URL = PUBLIC_HARNESS_WS_URL || "ws://127.0.0.1:8000/ws";

  let webcamStream = $state<MediaStream | null>(null);
  let webcamError = $state<string | null>(null);
  let videoEl = $state<HTMLVideoElement | null>(null);

  let ws = $state<WebSocket | null>(null);
  let wsState = $state<"idle" | "connecting" | "open" | "closed" | "error">("idle");
  let frameTimer = $state<number | null>(null);
  let framesSent = $state(0);
  let lastLatencyMs = $state<number | null>(null);

  let emotion = $state<Emotion>("neutral");
  let confidence = $state(0);
  let faceDetected = $state(false);
  let detectorLoaded = $state(false);
  let faceCropDataUrl = $state<string | null>(null);
  let recentPredictions = $state<{ ts: string; emotion: Emotion; conf: number }[]>([]);
  let emotionCounts = $state<Record<Emotion, number>>({
    neutral: 0, trust_relief: 0, sadness: 0, fear_anxiety: 0, confusion: 0, distrust: 0,
  });

  async function openWebcam() {
    webcamError = null;
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (err) {
      webcamError = err instanceof Error ? err.message : String(err);
    }
  }

  function attachStreamToVideo(node: HTMLVideoElement) {
    videoEl = node;
    if (webcamStream) node.srcObject = webcamStream;
    return {
      update(_props: unknown) {
        if (webcamStream && node.srcObject !== webcamStream) node.srcObject = webcamStream;
      },
      destroy() { videoEl = null; },
    };
  }

  $effect(() => {
    // Re-bind stream when it changes
    if (videoEl && webcamStream && videoEl.srcObject !== webcamStream) {
      videoEl.srcObject = webcamStream;
    }
  });

  function startSession() {
    if (!browser) return;
    if (ws) { stopSession(); }
    wsState = "connecting";
    const socket = new WebSocket(WS_URL);
    ws = socket;

    socket.onopen = () => {
      wsState = "open";
      socket.send(JSON.stringify({ type: "session_start", profile_id: "emotion-test" }));
      startFrameLoop(socket);
    };

    socket.onmessage = (ev) => {
      try { handleMessage(JSON.parse(ev.data)); }
      catch (e) { console.warn("emotion-test: bad WS message", e); }
    };

    socket.onclose = () => { wsState = "closed"; ws = null; stopFrameLoop(); };
    socket.onerror = () => { wsState = "error"; };
  }

  function stopSession() {
    if (ws) {
      try { ws.send(JSON.stringify({ type: "session_end" })); } catch {}
      ws.close();
      ws = null;
    }
    wsState = "idle";
    stopFrameLoop();
  }

  function startFrameLoop(socket: WebSocket) {
    const canvas = document.createElement("canvas");
    framesSent = 0;
    frameTimer = window.setInterval(() => {
      if (socket.readyState !== WebSocket.OPEN || !videoEl || !webcamStream) return;
      canvas.width = videoEl.videoWidth || 320;
      canvas.height = videoEl.videoHeight || 240;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(videoEl, 0, 0);
      const jpeg = canvas.toDataURL("image/jpeg", 0.6).split(",")[1];
      const sentAt = Date.now();
      socket.send(JSON.stringify({
        type: "video_frame",
        data: jpeg,
        timestamp: sentAt / 1000,
        _sent_at: sentAt,
      }));
      framesSent += 1;
    }, FRAME_INTERVAL_MS);
  }

  function stopFrameLoop() {
    if (frameTimer !== null) { clearInterval(frameTimer); frameTimer = null; }
  }

  function handleMessage(msg: Record<string, unknown>) {
    const t = msg.type;
    if (t === "face_detection") {
      faceDetected = Boolean(msg.detected);
      detectorLoaded = Boolean(msg.detector_loaded);
    } else if (t === "frame_debug") {
      const crop = msg.face_crop_data;
      if (typeof crop === "string") faceCropDataUrl = `data:image/jpeg;base64,${crop}`;
      else faceCropDataUrl = null;
    } else if (t === "emotion_update") {
      const e = msg.emotion;
      if (isEmotion(e)) {
        emotion = e;
        confidence = typeof msg.confidence === "number" ? msg.confidence : 0;
        emotionCounts[e] = (emotionCounts[e] ?? 0) + 1;
        recentPredictions = [
          { ts: new Date().toLocaleTimeString(), emotion: e, conf: confidence },
          ...recentPredictions,
        ].slice(0, 10);
      }
      const ts = msg.timestamp;
      if (typeof ts === "number") lastLatencyMs = Math.round(Date.now() - ts * 1000);
    }
  }

  // Debug flag toggles — these hit the model_service over plain HTTP.
  let forceLabel = $state<Emotion | "">("");
  let cycleLabels = $state(false);
  let logPredictions = $state(false);

  async function postDebug(payload: Record<string, unknown>) {
    const base = WS_URL.replace(/^ws/, "http").replace(/\/ws$/, "");
    try {
      await fetch(`${base}/api/v1/debug/emotion`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (e) {
      console.warn("emotion-test: debug POST failed", e);
    }
  }

  $effect(() => { postDebug({ force_label: forceLabel || null }); });
  $effect(() => { postDebug({ cycle_test_labels: cycleLabels }); });
  $effect(() => { postDebug({ log_predictions: logPredictions }); });

  const emotionsList: Emotion[] = ["neutral", "trust_relief", "sadness", "fear_anxiety", "confusion", "distrust"];
</script>

<svelte:head><title>emotion-test · empath_bot harness</title></svelte:head>

<main style="padding: 1.5rem; max-width: 1100px; margin: 0 auto; font-family: ui-sans-serif, system-ui, sans-serif;">
  <header style="display: flex; align-items: baseline; justify-content: space-between;">
    <h1 style="margin: 0; font-size: 1.4rem;">emotion-bot test harness</h1>
    <span style="opacity: 0.6; font-size: 0.85rem;">
      WS {wsState} · frames sent {framesSent}{lastLatencyMs !== null ? ` · ${lastLatencyMs}ms round-trip` : ""}
    </span>
  </header>

  <p style="opacity: 0.75; font-size: 0.9rem; margin-top: 0.3rem;">
    Bare harness for testing the face-detector → emotion-classifier pipeline. No chat, no
    LLM, no transcripts. The full app at <code>/</code> still works as normal.
  </p>

  <section style="display: grid; grid-template-columns: 1.5fr 1fr; gap: 1.2rem; margin-top: 1.2rem;">
    <!-- ── left: webcam + emotion display ── -->
    <div>
      <div style="border: 1px solid #333; border-radius: 8px; overflow: hidden; background: #000; aspect-ratio: 4/3;">
        {#if webcamStream}
          <!-- svelte-ignore a11y_media_has_caption -->
          <video use:attachStreamToVideo autoplay muted playsinline
                 style="width: 100%; height: 100%; object-fit: cover;"></video>
        {:else}
          <div style="display:flex; align-items:center; justify-content:center; height:100%; color:#888;">
            {webcamError ?? "Webcam not opened"}
          </div>
        {/if}
      </div>

      <div style="margin-top: 0.6rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
        {#if !webcamStream}
          <button onclick={openWebcam}>Open webcam</button>
        {/if}
        {#if wsState === "idle" || wsState === "closed" || wsState === "error"}
          <button onclick={startSession} disabled={!webcamStream}>Start session</button>
        {:else}
          <button onclick={stopSession}>Stop</button>
        {/if}
      </div>

      <!-- big emotion card -->
      <div style="
        margin-top: 1rem; padding: 1.2rem; border-radius: 10px;
        background: {EMOTION_COLOURS[emotion]}; color: #fff;
        display: flex; align-items: center; gap: 1rem;">
        <div style="font-size: 3rem; line-height: 1;">{EMOTION_EMOJI[emotion]}</div>
        <div style="flex: 1;">
          <div style="opacity: 0.7; font-size: 0.8rem; text-transform: uppercase;">Current emotion</div>
          <div style="font-size: 1.5rem; font-weight: 600;">{EMOTION_LABEL[emotion]}</div>
          <div style="opacity: 0.8; font-size: 0.85rem;">
            confidence {(confidence * 100).toFixed(1)}% · face_detected={String(faceDetected)} ·
            detector_loaded={String(detectorLoaded)}
          </div>
        </div>
        {#if faceCropDataUrl}
          <img src={faceCropDataUrl} alt="face crop" width="96" height="96"
               style="border-radius: 6px; object-fit: cover;" />
        {/if}
      </div>
    </div>

    <!-- ── right: debug controls + recent predictions ── -->
    <aside>
      <fieldset style="border: 1px solid #333; border-radius: 8px; padding: 0.7rem 0.9rem;">
        <legend style="padding: 0 0.4rem; opacity: 0.7; font-size: 0.85rem;">debug flags</legend>

        <label style="display: block; margin: 0.4rem 0;">
          Force label:
          <select bind:value={forceLabel}>
            <option value="">— off —</option>
            {#each emotionsList as e}
              <option value={e}>{EMOTION_LABEL[e]}</option>
            {/each}
          </select>
        </label>

        <label style="display: block; margin: 0.4rem 0;">
          <input type="checkbox" bind:checked={cycleLabels} />
          Cycle test labels
        </label>

        <label style="display: block; margin: 0.4rem 0;">
          <input type="checkbox" bind:checked={logPredictions} />
          Log predictions (INFO level)
        </label>

        <p style="opacity: 0.6; font-size: 0.75rem; margin: 0.3rem 0 0;">
          Toggles hit POST /api/v1/debug/emotion on the model service.
        </p>
      </fieldset>

      <fieldset style="border: 1px solid #333; border-radius: 8px; padding: 0.7rem 0.9rem; margin-top: 1rem;">
        <legend style="padding: 0 0.4rem; opacity: 0.7; font-size: 0.85rem;">running counts</legend>
        {#each emotionsList as e}
          <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin: 0.15rem 0;">
            <span>{EMOTION_EMOJI[e]} {EMOTION_LABEL[e]}</span>
            <span style="opacity: 0.7;">{emotionCounts[e]}</span>
          </div>
        {/each}
      </fieldset>

      <fieldset style="border: 1px solid #333; border-radius: 8px; padding: 0.7rem 0.9rem; margin-top: 1rem;">
        <legend style="padding: 0 0.4rem; opacity: 0.7; font-size: 0.85rem;">recent predictions</legend>
        {#if recentPredictions.length === 0}
          <div style="opacity: 0.6; font-size: 0.85rem;">none yet</div>
        {/if}
        {#each recentPredictions as p}
          <div style="font-size: 0.8rem; display: flex; justify-content: space-between; margin: 0.1rem 0;">
            <span>{p.ts} · {EMOTION_LABEL[p.emotion]}</span>
            <span style="opacity: 0.7;">{(p.conf * 100).toFixed(0)}%</span>
          </div>
        {/each}
      </fieldset>
    </aside>
  </section>
</main>
