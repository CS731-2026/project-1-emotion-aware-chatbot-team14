import {
  MAX_SPEECH_DURATION_MS,
  MIN_SPEECH_DURATION_MS,
  PRE_ROLL_MS,
  SAMPLE_RATE,
  SILENCE_HOLDOFF_MS,
  SPEECH_THRESHOLD,
  VAD_START_RATIO,
  VAD_STOP_RATIO,
  VAD_WINDOW_FRAMES,
} from "$lib/harness/types";

type BrowserVadCallbacks = {
  getMicStream: () => MediaStream | null;
  getSocket: () => WebSocket | null;
  onAudioLevel: (level: number) => void;
  onVadState: (state: string) => void;
  onTranscriptStatus: (status: string) => void;
  onDebug: (message: string) => void;
};

/**
 * Voice Activity Detection running on continuous PCM capture.
 *
 * The audio engine is *always* running. A small AudioWorklet streams 128-sample
 * frames back to the main thread; we keep a ring buffer of recent frames and a
 * sliding window of recent RMS levels. Each frame we ask "what fraction of the
 * last window of frames was above SPEECH_THRESHOLD?":
 *   - not in utterance + ratio >= VAD_START_RATIO  → enter utterance
 *   - in utterance     + ratio <  VAD_STOP_RATIO   → start silence holdoff
 *   - silence holdoff exceeds SILENCE_HOLDOFF_MS   → finalise + send
 *
 * Because capture is continuous, when an utterance starts we already have
 * PRE_ROLL_MS of audio in the buffer — we don't miss the first word.
 */
export class BrowserVadController {
  private audioContext: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private currentAudioLevel = 0;

  // Continuous PCM ring buffer. Each entry is a 128-sample frame.
  private frames: Float32Array[] = [];
  // Rolling RMS history matching `frames`.
  private levels: number[] = [];

  private utteranceActive = false;
  private utteranceStartedAt = 0;
  // Index into `frames` where the utterance is considered to start (already
  // backed off by PRE_ROLL_MS worth of frames).
  private utteranceStartFrameIdx = 0;
  private silenceStartedAt: number | null = null;

  // Cached because they're used per-frame to size the ring buffer.
  private static readonly FRAMES_PER_SECOND = SAMPLE_RATE / 128; // ≈ 125
  private static readonly PRE_ROLL_FRAMES = Math.ceil(
    (PRE_ROLL_MS / 1000) * BrowserVadController.FRAMES_PER_SECOND,
  );
  // When idle, cap the ring buffer at pre-roll + a small safety margin.
  // When in an utterance, the buffer grows up to MAX_SPEECH_DURATION_MS worth.
  private static readonly IDLE_BUFFER_FRAMES =
    BrowserVadController.PRE_ROLL_FRAMES + Math.ceil(BrowserVadController.FRAMES_PER_SECOND);
  private static readonly MAX_UTTERANCE_FRAMES = Math.ceil(
    (MAX_SPEECH_DURATION_MS / 1000) * BrowserVadController.FRAMES_PER_SECOND,
  );

  constructor(private callbacks: BrowserVadCallbacks) {}

  async start() {
    const micStream = this.callbacks.getMicStream();
    if (!micStream) {
      this.callbacks.onTranscriptStatus("Mic permission not available");
      return;
    }
    const socket = this.callbacks.getSocket();
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      this.callbacks.onTranscriptStatus("Mic enabled, waiting for harness connection");
      return;
    }

    if (this.workletNode) return; // already running

    const context = new AudioContext({ sampleRate: SAMPLE_RATE });
    try {
      await context.audioWorklet.addModule("/audio-vad-processor.js");
    } catch (err) {
      this.callbacks.onDebug(`AudioWorklet failed to load: ${err}`);
      await context.close();
      return;
    }

    const source = context.createMediaStreamSource(micStream);
    const worklet = new AudioWorkletNode(context, "vad-processor");
    worklet.port.onmessage = (event: MessageEvent<Float32Array>) => {
      this.onFrame(event.data);
    };
    source.connect(worklet);
    // Worklet doesn't need to play to speakers; not connecting to destination
    // avoids feedback. The worklet still runs.
    this.audioContext = context;
    this.sourceNode = source;
    this.workletNode = worklet;

    this.callbacks.onVadState("Listening");
    this.callbacks.onTranscriptStatus("Listening for your voice");
  }

  async stop(isListening: boolean) {
    if (this.workletNode) {
      this.workletNode.port.onmessage = null;
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
    if (this.audioContext) {
      try { await this.audioContext.close(); } catch { /* ignore */ }
      this.audioContext = null;
    }
    this.frames = [];
    this.levels = [];
    this.utteranceActive = false;
    this.silenceStartedAt = null;
    this.currentAudioLevel = 0;
    this.callbacks.onAudioLevel(0);
    if (!isListening) {
      this.callbacks.onTranscriptStatus("Mic idle");
      this.callbacks.onVadState("Mic idle");
    }
  }

  destroy() {
    void this.stop(false);
  }

  /** Receive a PCM frame from the worklet. Drives the windowed VAD state machine. */
  private onFrame(pcm: Float32Array) {
    // Compute RMS of this frame and update level history.
    let sumSq = 0;
    for (let i = 0; i < pcm.length; i++) sumSq += pcm[i] * pcm[i];
    const rms = Math.sqrt(sumSq / pcm.length);
    this.currentAudioLevel = rms;
    this.callbacks.onAudioLevel(rms);

    // Append to the ring buffer.
    this.frames.push(pcm);
    this.levels.push(rms);
    this.trimBuffersIfIdle();

    // Sliding-window ratio over the most recent VAD_WINDOW_FRAMES levels.
    const windowStart = Math.max(0, this.levels.length - VAD_WINDOW_FRAMES);
    let aboveCount = 0;
    for (let i = windowStart; i < this.levels.length; i++) {
      if (this.levels[i] > SPEECH_THRESHOLD) aboveCount++;
    }
    const windowSize = this.levels.length - windowStart;
    const ratio = windowSize > 0 ? aboveCount / windowSize : 0;

    if (!this.utteranceActive) {
      if (ratio >= VAD_START_RATIO) this.beginUtterance();
      else this.callbacks.onVadState(`Listening level=${rms.toFixed(5)}`);
      return;
    }

    // In an utterance: check both for trailing silence and for the safety cap.
    const utteranceMs = performance.now() - this.utteranceStartedAt;
    if (utteranceMs >= MAX_SPEECH_DURATION_MS) {
      this.finaliseUtterance("max duration reached");
      return;
    }

    if (ratio < VAD_STOP_RATIO) {
      if (this.silenceStartedAt === null) {
        this.silenceStartedAt = performance.now();
        this.callbacks.onVadState(`Silence detected, holding ${SILENCE_HOLDOFF_MS}ms`);
      } else if (performance.now() - this.silenceStartedAt >= SILENCE_HOLDOFF_MS) {
        this.finaliseUtterance("silence holdoff exceeded");
      }
    } else {
      this.silenceStartedAt = null;
      this.callbacks.onVadState(`Recording speech level=${rms.toFixed(5)}`);
    }
  }

  /** While idle, keep only the most recent pre-roll worth of frames + a small margin. */
  private trimBuffersIfIdle() {
    if (this.utteranceActive) return;
    while (this.frames.length > BrowserVadController.IDLE_BUFFER_FRAMES) {
      this.frames.shift();
      this.levels.shift();
    }
  }

  private beginUtterance() {
    this.utteranceActive = true;
    this.utteranceStartedAt = performance.now();
    // Pre-roll: anchor "start" PRE_ROLL_FRAMES before the current frame,
    // clamped to the start of the buffer.
    this.utteranceStartFrameIdx = Math.max(
      0,
      this.frames.length - BrowserVadController.PRE_ROLL_FRAMES,
    );
    this.silenceStartedAt = null;
    this.callbacks.onVadState("Recording speech");
    this.callbacks.onDebug(`utterance begin (pre-roll=${PRE_ROLL_MS}ms)`);
  }

  private finaliseUtterance(reason: string) {
    if (!this.utteranceActive) return;
    const durationMs = performance.now() - this.utteranceStartedAt;
    const captured = this.frames.slice(this.utteranceStartFrameIdx);
    const totalSamples = captured.reduce((n, f) => n + f.length, 0);
    const capturedMs = (totalSamples / SAMPLE_RATE) * 1000;

    this.utteranceActive = false;
    this.silenceStartedAt = null;
    // Reset the ring buffer so the next utterance starts clean (pre-roll
    // refills naturally as new frames arrive).
    this.frames = [];
    this.levels = [];
    this.callbacks.onVadState(`Stopping speech clip (${reason})`);

    if (capturedMs < MIN_SPEECH_DURATION_MS) {
      const msg = `Discarded short clip (${(capturedMs / 1000).toFixed(1)}s, reason=${reason})`;
      this.callbacks.onVadState(msg);
      this.callbacks.onDebug(msg);
      return;
    }

    const socket = this.callbacks.getSocket();
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      this.callbacks.onDebug("utterance ready but WebSocket not open; dropping");
      return;
    }

    const merged = concatFrames(captured, totalSamples);
    const wavBase64 = encodeWavBase64(merged, SAMPLE_RATE);
    socket.send(JSON.stringify({
      type: "audio_chunk",
      data: wavBase64,
      format: "wav",
      timestamp: Date.now() / 1000,
      duration_ms: Math.round(capturedMs),
      rms_level: Number(this.currentAudioLevel.toFixed(6)),
    }));
    this.callbacks.onTranscriptStatus(
      `Sent speech clip ${(capturedMs / 1000).toFixed(1)}s (utterance ${(durationMs / 1000).toFixed(1)}s)`,
    );
    this.callbacks.onVadState("Clip sent, waiting for transcript");
    this.callbacks.onDebug(
      `sent clip ${(capturedMs / 1000).toFixed(1)}s, ${wavBase64.length} base64 chars`,
    );
  }
}

function concatFrames(frames: Float32Array[], totalSamples: number): Float32Array {
  const out = new Float32Array(totalSamples);
  let offset = 0;
  for (const f of frames) {
    out.set(f, offset);
    offset += f.length;
  }
  return out;
}

/** Encode a mono Float32 PCM buffer as a base64 16-bit PCM WAV blob. */
function encodeWavBase64(samples: Float32Array, sampleRate: number): string {
  const byteLength = 44 + samples.length * 2;
  const buffer = new ArrayBuffer(byteLength);
  const view = new DataView(buffer);

  // RIFF header
  view.setUint32(0, 0x46464952, true);            // "RIFF"
  view.setUint32(4, byteLength - 8, true);
  view.setUint32(8, 0x45564157, true);            // "WAVE"
  // fmt chunk
  view.setUint32(12, 0x20746d66, true);           // "fmt "
  view.setUint32(16, 16, true);                   // PCM chunk size
  view.setUint16(20, 1, true);                    // PCM format
  view.setUint16(22, 1, true);                    // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);       // byte rate (sampleRate * blockAlign)
  view.setUint16(32, 2, true);                    // block align (channels * bytesPerSample)
  view.setUint16(34, 16, true);                   // bits per sample
  // data chunk
  view.setUint32(36, 0x61746164, true);           // "data"
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}
