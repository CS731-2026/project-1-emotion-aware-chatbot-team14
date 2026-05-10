import {
  AUDIO_CHUNK_SIZE,
  MAX_SPEECH_DURATION_MS,
  MIN_SPEECH_DURATION_MS,
  SILENCE_DURATION_MS,
  SPEECH_THRESHOLD,
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
 * Voice Activity Detection controller that runs entirely in the browser.
 *
 * Uses Web Audio API's AnalyserNode to compute RMS level every 100 ms. When
 * level exceeds SPEECH_THRESHOLD it starts a MediaRecorder; when silence is
 * detected for SILENCE_DURATION_MS it stops, encodes the clip to base64, and
 * sends an audio_chunk message over the WebSocket to the model service.
 */
export class BrowserVadController {
  private mediaRecorder: MediaRecorder | null = null;
  private audioContext: AudioContext | null = null;
  private audioAnalyser: AnalyserNode | null = null;
  private vadInterval: ReturnType<typeof setInterval> | null = null;
  private speechStopTimeout: ReturnType<typeof setTimeout> | null = null;
  private maxSpeechTimeout: ReturnType<typeof setTimeout> | null = null;
  private recordingStartedAt = 0;
  private currentAudioLevel = 0;

  constructor(private callbacks: BrowserVadCallbacks) {}

  start() {
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

    if (this.mediaRecorder || this.vadInterval) return;

    if (!this.ensureAudioAnalyser(micStream)) {
      this.callbacks.onTranscriptStatus("Mic analyser could not start");
      return;
    }

    this.callbacks.onTranscriptStatus("Listening with browser VAD");
    this.callbacks.onVadState(`Listening (threshold ${SPEECH_THRESHOLD})`);
    this.callbacks.onDebug("VAD loop started");

    this.vadInterval = setInterval(() => this.tick(), 100);
  }

  stop(isListening: boolean) {
    if (this.vadInterval) {
      clearInterval(this.vadInterval);
      this.vadInterval = null;
    }

    this.stopSpeechRecording("mic disabled");

    if (this.audioContext) {
      this.audioContext.close();
    }
    this.audioContext = null;
    this.audioAnalyser = null;
    this.mediaRecorder = null;
    this.currentAudioLevel = 0;
    this.callbacks.onAudioLevel(0);

    if (!isListening) {
      this.callbacks.onTranscriptStatus("Mic idle");
      this.callbacks.onVadState("Mic idle");
    }
  }

  destroy() {
    this.stop(false);
  }

  /** Called every 100 ms: measure audio level and drive the recording state machine. */
  private tick() {
    this.currentAudioLevel = this.getAudioLevel();
    this.callbacks.onAudioLevel(this.currentAudioLevel);
    const speaking = this.currentAudioLevel > SPEECH_THRESHOLD;

    if (speaking && !this.mediaRecorder) {
      this.startSpeechRecording();
      return;
    }

    if (speaking && this.mediaRecorder) {
      if (this.speechStopTimeout) {
        clearTimeout(this.speechStopTimeout);
        this.speechStopTimeout = null;
      }
      this.callbacks.onVadState(`Recording speech level=${this.currentAudioLevel.toFixed(5)}`);
      return;
    }

    if (!speaking && this.mediaRecorder && !this.speechStopTimeout) {
      this.callbacks.onVadState(
        `Silence detected, stopping after ${(SILENCE_DURATION_MS / 1000).toFixed(1)}s`,
      );
      this.speechStopTimeout = setTimeout(() => {
        this.stopSpeechRecording("silence detected");
      }, SILENCE_DURATION_MS);
    } else if (!speaking && !this.mediaRecorder) {
      this.callbacks.onVadState(`Listening level=${this.currentAudioLevel.toFixed(5)}`);
    }
  }

  /** Create AudioContext + AnalyserNode from the mic stream (idempotent). */
  private ensureAudioAnalyser(micStream: MediaStream) {
    if (this.audioAnalyser) return true;

    const context = new AudioContext({ sampleRate: 16000 });
    const source = context.createMediaStreamSource(micStream);
    const analyser = context.createAnalyser();
    analyser.fftSize = AUDIO_CHUNK_SIZE;
    source.connect(analyser);
    this.audioContext = context;
    this.audioAnalyser = analyser;
    return true;
  }

  /** Compute the RMS amplitude of the current audio frame from the analyser. */
  private getAudioLevel() {
    if (!this.audioAnalyser) return 0;

    const samples = new Float32Array(this.audioAnalyser.fftSize);
    this.audioAnalyser.getFloatTimeDomainData(samples);

    let sum = 0;
    for (const sample of samples) {
      sum += sample * sample;
    }
    return Math.sqrt(sum / samples.length);
  }

  /** Start a MediaRecorder session; on stop, encode the blob and send it over WS. */
  private startSpeechRecording() {
    const micStream = this.callbacks.getMicStream();
    const socket = this.callbacks.getSocket();
    if (!micStream || this.mediaRecorder || !socket || socket.readyState !== WebSocket.OPEN) return;

    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";
    const recorder = new MediaRecorder(micStream, { mimeType });
    this.mediaRecorder = recorder;
    this.recordingStartedAt = performance.now();
    this.callbacks.onVadState("Recording speech");
    this.callbacks.onDebug(`speech start level=${this.currentAudioLevel.toFixed(5)}`);

    recorder.ondataavailable = async (event) => {
      if (!event.data || event.data.size === 0) return;
      const activeSocket = this.callbacks.getSocket();
      if (!activeSocket || activeSocket.readyState !== WebSocket.OPEN) return;

      const durationMs = performance.now() - this.recordingStartedAt;
      if (durationMs < MIN_SPEECH_DURATION_MS) {
        const state = `Discarded short clip (${(durationMs / 1000).toFixed(1)}s)`;
        this.callbacks.onVadState(state);
        this.callbacks.onDebug(state);
        return;
      }

      const data = await blobToBase64(event.data);
      activeSocket.send(JSON.stringify({
        type: "audio_chunk",
        data,
        timestamp: Date.now() / 1000,
        duration_ms: Math.round(durationMs),
        rms_level: Number(this.currentAudioLevel.toFixed(6)),
      }));
      this.callbacks.onTranscriptStatus(`Sent speech clip ${(durationMs / 1000).toFixed(1)}s to harness`);
      this.callbacks.onVadState("Clip sent, waiting for transcript");
      this.callbacks.onDebug(`sent clip ${(durationMs / 1000).toFixed(1)}s (${data.length} base64 chars)`);
    };

    recorder.onstop = () => {
      this.mediaRecorder = null;
      this.recordingStartedAt = 0;
      if (this.maxSpeechTimeout) {
        clearTimeout(this.maxSpeechTimeout);
        this.maxSpeechTimeout = null;
      }
    };

    recorder.start();
    this.maxSpeechTimeout = setTimeout(() => {
      this.stopSpeechRecording("max duration reached");
    }, MAX_SPEECH_DURATION_MS);
  }

  /** Stop the active MediaRecorder, clearing both the silence and max-duration timers. */
  private stopSpeechRecording(reason: string) {
    if (this.speechStopTimeout) {
      clearTimeout(this.speechStopTimeout);
      this.speechStopTimeout = null;
    }
    if (this.maxSpeechTimeout) {
      clearTimeout(this.maxSpeechTimeout);
      this.maxSpeechTimeout = null;
    }
    if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
      const state = `Stopping speech clip (${reason})`;
      this.callbacks.onVadState(state);
      this.callbacks.onDebug(state);
      this.mediaRecorder.stop();
    }
  }
}

async function blobToBase64(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;

  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }

  return btoa(binary);
}
