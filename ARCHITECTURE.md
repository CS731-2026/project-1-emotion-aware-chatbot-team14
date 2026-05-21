# Architecture Guide

A walkthrough of how the system is structured and how data flows through it — including exact function call chains, integration points for the emotion model, and where to expand LLM reasoning.

---

## The Big Picture

This system is an **emotion-aware empathy bot**. The core idea: read the user's face via webcam, classify their emotional state in real time, and combine that signal with a live speech transcript to condition the LLM's responses — two separate inputs converging at LLM Reasoning.

Three services run together:

```
┌─────────────────────────────────────────────────────────────┐
│  Browser                                                    │
│  SvelteKit frontend  (localhost:5173)                       │
│    │  HTTP (fetch)                WebSocket (/ws)           │
│    ▼                              │                         │
│  Express backend (localhost:3001) │                         │
│    │  HTTP (fetch)                │                         │
│    ▼                              ▼                         │
│  FastAPI model service (localhost:8000)                     │
│    └── YOLOv8 face detector                                 │
│    └── Emotion model — EmpathBotV1 (via models.yaml)        │
│    │       placeholder / resnet18 also selectable           │
│    └── whisper.cpp STT                                      │
│    └── LLM (OpenAI / Gemini / Ollama / Anthropic)           │
└─────────────────────────────────────────────────────────────┘
```

**Key constraint**: the browser never calls Express or the model service directly. All HTTP calls go through SvelteKit server-side. The only exception is the WebSocket, which the browser opens directly to the model service at `ws://localhost:8000/ws`.

---

## Three Parallel Data Flows

Per the architecture spec, two inputs feed LLM Reasoning independently:
- **Emotional context** (Face → YOLO → Emotional Affect Model → EmotionalReasoningAgent)
- **Transcript with timestamps** (Mic → VAD → STT → transcript buffer)

Both are injected as separate system messages into `LLMReasoningAgent.reason()`. The exact prompt engineering strategy for combining them is **TBD**.

### Flow 1 — Video → Face Detection → Emotion Buffer

This is the continuous stream that populates the emotional context used by the LLM.

```
frontend/src/routes/+page.svelte
  captureVideoFrame()             — canvas.drawImage(videoEl) → toDataURL("image/jpeg")
  ws.send({type:"video_frame", data: base64JPEG, timestamp})
  [every FRAME_INTERVAL_MS = 200 ms — see frontend/src/lib/harness/types.ts]
        │
        ▼  WebSocket
application/model_service/ws/handler.py
  handle_websocket()              — main WS loop, dispatches on msg["type"]
    │  msg_type == "video_frame"
    │
    ├─ base64.b64decode(msg["data"])
    │  np.frombuffer(...) → cv2.imdecode() → frame_bgr (H×W×3 uint8)
    │
    ├─ core/face_detector.py  FaceDetector.detect_best(frame_bgr)
    │    self._model(frame_bgr, conf=0.35, device=self.device)
    │      └─ ultralytics YOLO inference (YOLOv8-face, HuggingFace cached)
    │    supervision.Detections.from_ultralytics(results[0])
    │    np.argmax(detections.confidence) → best_idx
    │    detections.xyxy[best_idx] → box  (clamped to frame bounds)
    │    frame_bgr[y1:y2, x1:x2] → face_crop (BGR uint8)
    │    returns (face_crop, box_xyxy) or (None, None)
    │
    │  ┌──────────────────────────────────────────────────────────────────────┐
    │  │  EMOTION MODEL INVOCATION                                            │
    │  │  ws/handler.py  pick_emotion(face_crop, emotion_model, detected)     │
    │  │                                                                      │
    │  │  Reads core/debug_flags.py::emotion at runtime. Priority:           │
    │  │    1. force_label       → pin a label (debug)                       │
    │  │    2. cycle_test_labels → step through EMOTIONS on a timer (debug)  │
    │  │    3. emotion_model.predict(face_crop)                              │
    │  │    4. ("neutral", 0.5)  fallback                                    │
    │  │                                                                      │
    │  │  Model selection (loaded once at startup):                          │
    │  │    EMOTION_MODEL_ID=<id>  → resolved via models.yaml (preferred)    │
    │  │    EMOTION_VARIANT=<name> → legacy fallback                         │
    │  │  See "Integration Points" below for adding a new variant.           │
    │  └──────────────────────────────────────────────────────────────────────┘
    │
    ├─ core/emotion/buffer.py  EmotionBuffer.update(emotion, confidence, timestamp)
    │    deque(maxlen=10).append(EmotionObservation(...))
    │    [rolling window — oldest observation evicted automatically]
    │
    └─ ws.send({type:"face_detection", detected, timestamp})
       ws.send({type:"frame_debug", box, timings_ms, image_data, face_crop_data})
       ws.send({type:"emotion_update", emotion, confidence, timestamp})
              │
              ▼  WebSocket
        frontend: updates UI background colour via EMOTION_COLOURS map (types.ts)
```

**Face detector internals summary:**
- Model: `arnabdhar/YOLOv8-Face-Detection` (auto-downloaded from HuggingFace on first run, cached to `model_service/models/`)
- Device: MPS (Apple Silicon) → CUDA → CPU, auto-selected at init
- Confidence threshold: 0.35 (hardcoded in `FaceDetector.CONF_THRESHOLD`)
- Output: highest-confidence face crop only (`detect_best`), not all faces

---

### Flow 2 — Microphone → VAD → STT → Transcript Buffer

This flow captures what the user says and stores it as timestamped transcript segments. The transcript buffer is passed to `EmotionalReasoningAgent` when a chat message is sent.

```
frontend/src/lib/harness/browserVad.ts
  BrowserVadController.start()
    ensureAudioAnalyser(micStream)
      AudioContext({ sampleRate: 16000 })
      createMediaStreamSource(micStream) → AnalyserNode (fftSize = AUDIO_CHUNK_SIZE = 512)
    setInterval(tick, 100ms)            — VAD polling loop

  tick() [every 100 ms]
    getAudioLevel()                     — RMS of AnalyserNode.getFloatTimeDomainData()
    if level > SPEECH_THRESHOLD (0.00075):
      startSpeechRecording()
        MediaRecorder(micStream, {mimeType: "audio/webm;codecs=opus"})
        recorder.start()
        maxSpeechTimeout = setTimeout(stop, MAX_SPEECH_DURATION_MS = 30s)
    if silence > SILENCE_DURATION_MS (600ms):
      stopSpeechRecording()
        recorder.stop()
          ondataavailable(event):
            if duration < MIN_SPEECH_DURATION_MS (1200ms): discard clip
            blobToBase64(event.data)    — ArrayBuffer → btoa()
            ws.send({type:"audio_chunk", data: base64WebM, timestamp, duration_ms, rms_level})
        │
        ▼  WebSocket
application/model_service/ws/handler.py
  handle_websocket()
    msg_type == "audio_chunk"
    asyncio.create_task(process_audio_chunk(...))   — non-blocking; event loop stays free

  process_audio_chunk(websocket, session, stt, chunk_count, data, timestamp)
    decode_browser_audio_to_numpy(data)
      base64.b64decode(data) → writes to tmp/chunk.webm
      subprocess ffmpeg: webm → s16le PCM (mono, 16 kHz)
      np.frombuffer(pcm_bytes, dtype=np.int16) / 32768.0  → float32 [-1.0, 1.0]
    [runs in asyncio.to_thread → worker thread, not blocking event loop]

    stt.transcribe(audio_np)
      └─ core/stt/whisper_cpp.py  WhisperCppTranscriptionService.transcribe()
           [or core/stt/whisper_faster.py depending on STT_ENGINE env var]
           returns (text, language, confidence)
    [runs in asyncio.to_thread → worker thread]

    session.transcript_buffer.append(TranscriptSegment(text, timestamp))
    [capped at 20 segments — oldest trimmed when limit exceeded]

    ws.send({type:"transcript_chunk", text, timestamp})
    ws.send({type:"audio_debug", timings_ms, stt_error, ...})
         │
         ▼  WebSocket
   frontend: TranscriptHistory.svelte displays the text
```

**STT internals summary:**
- Default engine: `whisper-cpp` (set via `STT_ENGINE` env var)
- Audio format required: float32 PCM, mono, 16 kHz — this is why ffmpeg is needed (browser records WebM/Opus)
- Both STT backends implement `TranscriptionService.transcribe(audio_np) → (text, lang, confidence)`
- The transcript buffer is accessible at `session.transcript_buffer[-20:]` — passed to both `EmotionalReasoningAgent.analyse()` and directly to `LLMReasoningAgent.reason()` as a separate system message input

---

### Flow 3 — Chat Message → Emotion Context → LLM → Response

This is the user-facing request/response cycle. It reads from both the emotion buffer (Flow 1) and the transcript buffer (Flow 2) to produce a contextualised LLM reply.

```
frontend/src/lib/components/ChatInput.svelte
  user submits text
  api.sendChat(text)              — src/lib/api.ts
    fetch("http://localhost:3001/api/v1/chat", {method:"POST", body:{text}})
         │
         ▼  HTTP POST /api/v1/chat
application/backend/src/routes/chat.router.ts
  POST /  handler
    req.session.profileId         — 401 if no active profile
    profileStore.getHistory(profileId)   — src/lib/profileStore.ts
      reads data/profiles/<uuid>.json → file.messages[]
    windowed = fullHistory.slice(-(HISTORY_WINDOW * 2))  [last 20 messages]
    harnessHistory = windowed.map(m => ({role: m.role==="agent"?"assistant":m.role, content}))
    fetch(`${MODEL_SERVICE_URL}/api/v1/chat`, {profile_id, message: text, history: harnessHistory})
         │
         ▼  HTTP POST /api/v1/chat
application/model_service/routers/chat.py
  chat(body: ChatRequest, request: Request)
    ws.handler.get_session(body.profile_id)
      _sessions.get(profile_id)   — in-memory dict linking WS session → HTTP request
                                  — returns HarnessSession or None

    session.emotion_buffer.history()      — core/emotion/buffer.py
      list(deque)  → list[EmotionObservation(emotion, confidence, timestamp)]

    emotion_agent.analyse(emotion_observations, transcript_segments[-20:])
    └─ core/emotional_reasoning_agent.py  EmotionalReasoningAgent.analyse()
         statistics.mode(obs.emotion for obs in emotion_observations)  → dominant
         max(timestamps) - min(timestamps)  → duration_seconds
         returns: "The user appears to be feeling {dominant} (~{N}s).
                   Calibrate tone accordingly without referencing this directly."

    llm_agent.reason(body.message, emotional_context, history, transcript_segments)
    └─ core/llm/reasoning_agent.py  LLMReasoningAgent.reason()
         messages = []
         messages.append({role:"system", content: SYSTEM_PROMPT})         # empathy bot persona (TBD)
         prior = [m for m in history if m["role"] in ("user","assistant")]
         messages.extend(prior[-history_window:])                          # last 10 turns
         messages.append({role:"system", content: emotional_context})      # ← Emotional Affect Model output
         messages.append({role:"system", content: transcript_context})     # ← STT output (separate input)
         messages.append({role:"user", content: message})                  # current turn
         self._llm.chat(messages)
         └─ core/llm/openai.py  OpenAIProvider.chat()
              self._client.chat.completions.create(model=self._model, messages=messages)
              → response.choices[0].message.content
              returns: assistant reply string

    returns ChatResponse(response=reply)
         │
         ▼  HTTP response
application/backend/src/routes/chat.router.ts
    profileStore.appendMessage(profileId, userMsg)   # write user turn to disk
    profileStore.appendMessage(profileId, agentMsg)  # write agent reply to disk
    res.json({response})
         │
         ▼  HTTP response
frontend: ChatHistory.svelte renders the new messages
```

---

## Integration Points — What to Build Next

### 1. Adding a new emotion model variant

The pipeline already runs a real model (`EmpathBotV1`) by default via the registry. To add another variant — e.g. a different architecture, ensemble, or a teammate's hand-trained checkpoint:

**Step 1 — Implement `EmotionModel` in a new file:**
```
application/model_service/core/emotion/<your_model_name>.py
```
Must implement the ABC from `core/emotion/base.py`:
```python
def predict(self, face_bgr: np.ndarray) -> tuple[str, float]:
    # face_bgr: uint8 BGR crop of the detected face, any size
    # returns: (emotion_label, confidence) where label ∈ EMOTIONS
    #   EMOTIONS = ['neutral', 'trust_relief', 'sadness',
    #               'fear_anxiety', 'confusion', 'distrust']
```

The model **is the source of truth** for label semantics. If your checkpoint stores `class_names`, assert it matches `EMOTIONS` at load time and fail loud on mismatch (see `core/emotion/empathbot.py` for the pattern).

**Step 2 — Register it in the factory:**
```python
# core/emotion/factory.py
if variant == "your_model_name":
    from .your_model_name import YourEmotionModel
    return YourEmotionModel(checkpoint_path=checkpoint_path, device=config.EMOTION_DEVICE)
```

**Step 3 — Add a registry entry:**
```yaml
# application/model_service/models.yaml
models:
  your_model_id:
    path:    models/your_model/your_checkpoint.pth
    variant: your_model_name
```

**Step 4 — Select it in `.env`:**
```
EMOTION_MODEL_ID=your_model_id
```

`pick_emotion()` already calls `emotion_model.predict(face_crop)` whenever a model is loaded — no other changes needed. The face crop arrives as BGR uint8 from the face detector; resize and normalise inside `predict()` to match the model's expected input.

---

### 2. LLM Reasoning Expansion

There are three layers where reasoning can be expanded, ordered from easiest to most impactful:

#### Layer A — `EmotionalReasoningAgent.analyse()` in `core/emotional_reasoning_agent.py`

**Current:** uses `statistics.mode` over raw emotion labels, ignores confidence scores, ignores transcript segments entirely.

**Expand to:**
- Weight recent observations more heavily (recency-biased mode or exponential moving average)
- Factor in confidence: low-confidence observations should count less
- Use `transcript_segments` (currently passed in but unused): scan for verbal cues to reinforce the emotional signal
- Produce richer context: include confidence, duration, trend (improving/worsening), and key transcript phrases

Example of a richer output:
```
"The user has appeared frustrated (angry/sad) for ~18s (confidence 0.82, worsening).
They recently said: 'I don't understand why this works'.
Slow down. Offer a simpler framing before continuing."
```

This string is injected verbatim as a `system` role message directly before the user's turn — so more specific language translates directly into more tailored LLM behaviour.

#### Layer B — `LLMReasoningAgent.reason()` in `core/llm/reasoning_agent.py`

**Current:** static 10-message history window; emotional context is a single sentence injected as a system message.

**Expand to:**
- Dynamically adjust `history_window` based on emotional state
- Instead of a single context string, pass structured data (emotion, confidence, duration, transcript excerpt) and let the prompt template format it
- Add a second `EmotionalReasoningAgent` call that uses the full transcript history, not just the 20-segment window

#### Layer C — `SYSTEM_PROMPT` in `core/llm/reasoning_agent.py`

**Current:** a single static string loaded at module import time.

**Expand to:**
- A dict of per-emotion system prompts selected based on the dominant emotion
- Or a template that varies explanation depth, tone directives, and example-giving style by emotional state
- The prompt is sent as the first message every turn — it's the highest-leverage place to shape LLM behaviour

---

## Component State Map

```
ws/handler.py: handle_websocket()           — thin dispatcher only
  │
  ├─ "video_frame" → on_video_frame()       — composition point; all steps explicit here
  │    ws/video.py: detect_from_message()
  │      decode_frame()                         decode base64 JPEG → BGR numpy
  │      run_face_detection()                   YOLO → face_crop, box, annotated frame
  │        core/face_detector.py: detect_best() [WORKING]
  │    ws/handler.py: pick_emotion()            [WORKING — calls real model]
  │      reads core/debug_flags.py::emotion (force/cycle/log override)
  │      emotion_model.predict(face_crop)       only invocation point in the codebase
  │    core/emotion/buffer.py: EmotionBuffer.update()   [WORKING]
  │    _send_frame_messages()                   face_detection + frame_debug + emotion_update
  │
  └─ "audio_chunk" → on_audio_chunk()
       ws/audio.py: process_audio_chunk()
         decode_browser_audio_to_numpy()        [WORKING] ffmpeg WebM → float32 PCM
         stt.transcribe(audio_np)               [WORKING] → (text, lang, conf)
         session.transcript_buffer.append()     [WORKING]

routers/chat.py: chat()
  │
  ├─ ws/session.py: get_session(profile_id)
  ├─ emotion_agent.analyse()    core/emotional_reasoning_agent.py  [BASIC → EXPAND]
  │    uses: emotion_buffer.history()
  │    ignores: transcript_buffer              ← low-hanging fruit: use this
  │
  └─ llm_agent.reason()         core/llm/reasoning_agent.py        [WORKING → EXPAND]
       assembles prompt → LLMProvider.chat()
       └─ openai.py / ollama.py / anthropic.py (stub)
```

---

## Service Internals

### Frontend (`application/frontend/`)

Built with SvelteKit + Svelte 5. One page: `src/routes/+page.svelte`.

**Svelte 5 runes** are used throughout — `$state`, `$derived`, `$effect`, `$props()`. No legacy `export let` or `on:*` event syntax.

```
src/
├── routes/
│   ├── +page.svelte       Main UI (chat, webcam, debug panel, transcript)
│   └── +page.server.ts    Server-side load (fetches session + profiles + history)
├── lib/
│   ├── api.ts             Typed fetch wrappers for all backend endpoints
│   ├── harness/
│   │   ├── browserVad.ts  VAD controller (mic → WS audio_chunk messages)
│   │   └── types.ts       Constants (FRAME_INTERVAL_MS, SPEECH_THRESHOLD, etc.) + emotion types
│   └── components/
│       ├── ChatHistory.svelte       Renders conversation turns
│       ├── ChatInput.svelte         Text input + submit
│       ├── WebcamPreview.svelte     Shows annotated webcam feed (with face box overlay)
│       ├── DebugDashboard.svelte    Live timing stats from frame_debug + audio_debug WS messages
│       ├── TranscriptHistory.svelte Shows STT transcript chunks
│       ├── ProfileModal.svelte      Profile creation + selection
│       ├── SpeakingCircle.svelte    Audio level visualiser
│       ├── SideNotes.svelte
│       └── Button.svelte
```

### Backend (`application/backend/`)

Express 4 + TypeScript. Thin proxy and session/history owner.

```
src/
├── index.ts               Server start; handles EADDRINUSE cleanly
├── app.ts                 CORS (hardcoded localhost:5173), express-session, route mount
├── config/env.ts          Single source of truth for all env vars
├── lib/profileStore.ts    File-based profile + message store (data/profiles/)
├── middleware/errorHandler.ts  Catches next(err) calls; returns JSON error response
└── routes/
    ├── index.ts               Mounts all routers under /api/v1/
    ├── chat.router.ts         POST / → proxies to model service; fallback on failure
    ├── profiles.router.ts     GET / list, POST / create, POST /:id/select → sets session
    ├── history.router.ts      GET / read history, POST / append message
    └── session.router.ts      GET / → returns {profileId} from session
```

**Profile storage:** each profile gets `data/profiles/<uuid>.json` → `{profile, messages[]}`. `data/profiles/index.json` is a flat list of profile metadata only. The index is read on every list/lookup call (no in-memory cache).

### Model Service (`application/model_service/`)

FastAPI + Python. All ML work happens here.

```
app.py           FastAPI app creation, lifespan (loads all ML components), WS + router mounting
config.py        All env vars (STT_ENGINE, EMOTION_MODEL_ID, EMOTION_VARIANT, LLM_PROVIDER, EMOTION_CYCLE_TEST_LABELS, EMOTION_FORCE_LABEL, …)
                 Plus load_model_registry() — reads models.yaml
models.yaml      Local model registry (id → path + variant)
main.py          Uvicorn entry point
routers/
└── chat.py      POST /api/v1/chat — the only HTTP route; reads WS session state, calls LLM
ws/
├── handler.py   Dispatcher + composition — routes WS messages; on_video_frame composes the pipeline
│                  pick_emotion() lives here — the only point emotion_model.predict() is called
├── session.py   HarnessSession dataclass, _sessions store, get_session, emit_debug
├── audio.py     decode_browser_audio_to_numpy (ffmpeg) + process_audio_chunk (STT pipeline)
└── video.py     Pure frame utilities — decode_frame, run_face_detection, encode_jpeg_b64,
│                  detect_from_message, FrameDetectionResult — no emotion logic
└── protocol.py  Dataclass definitions for all WS message types (documentation reference)
core/
├── __init__.py                  Re-exports debug_flags for `from core import debug_flags`
├── debug_flags.py               Mutable runtime flags (cycle/force/log) seeded from .env
├── face_detector.py             YOLOv8 face detector (HuggingFace, auto-downloaded + cached)
├── emotional_reasoning_agent.py EmotionObservation[] + transcript[] → context string
├── emotion/
│   ├── base.py        EmotionModel ABC + EMOTIONS list (EmpathBot 6-class)
│   ├── buffer.py      EmotionBuffer (deque, window=10) + EmotionObservation dataclass
│   ├── factory.py     create_emotion_model() — checks EMOTION_MODEL_ID then EMOTION_VARIANT
│   ├── placeholder.py Returns random emotion (kept for fallback/testing)
│   ├── resnet18.py    Vanilla torchvision ResNet18 + Linear head
│   └── empathbot.py   EmpathBotV1 (EfficientNet-B2 / ResNet18+SE) — currently loaded by default
├── llm/
│   ├── base.py            LLMProvider ABC + Message TypedDict
│   ├── openai.py          OpenAI Chat Completions (working)
│   ├── anthropic.py       Stub — raises NotImplementedError
│   ├── ollama.py          Ollama local LLM (implemented, untested)
│   ├── factory.py         create_llm(provider, model)
│   └── reasoning_agent.py LLMReasoningAgent — assembles full message list per turn
└── stt/
    ├── base.py              TranscriptionService ABC
    ├── whisper_cpp.py       whisper.cpp backend (default; requires WHISPER_CPP_DIR)
    ├── whisper_faster.py    faster-whisper backend (CPU-friendly alternative)
    └── factory.py           create_stt(engine, model)
```

**Lifespan pattern:** all ML components load once at startup inside `app.py:lifespan()` and are stored on `app.state`. Each loads independently — if YOLOv8 weights are missing the face detector fails gracefully; STT, emotion model, and LLM still load. Check `GET /health` to see which components are live.

**Factory pattern:** every swappable ML backend (LLM, STT, emotion model) follows the same three-file pattern: `base.py` (ABC) → `<name>.py` (implementation) → `factory.py` (`create_X()` selects and instantiates). To add a new backend, add a file and a branch in the factory — nothing else changes.

---

## WebSocket Protocol

**Inbound (browser → model service):**

| type | key payload fields | sent when |
|---|---|---|
| `session_start` | `profile_id: str` | browser opens WS |
| `video_frame` | `data: base64 JPEG`, `timestamp: float` | every 200 ms (`FRAME_INTERVAL_MS`) |
| `audio_chunk` | `data: base64 WebM`, `timestamp: float`, `duration_ms`, `rms_level` | after VAD detects speech |
| `session_end` | — | page unload |

**Outbound (model service → browser):**

| type | key payload fields | triggered by |
|---|---|---|
| `connection_ack` | — | WS connect |
| `message_ack` | `message_type` | every inbound message |
| `harness_status` | `face_detector_loaded`, `stt_loaded`, `emotion_model_loaded`, `llm_loaded`, … | `session_start` |
| `face_detection` | `detected: bool`, `timestamp` | every `video_frame` |
| `frame_debug` | `box`, `timings_ms` (decode/yolo/jpeg_encode), `image_data` (annotated JPEG), `face_crop_data` | every `video_frame` |
| `emotion_update` | `emotion: str`, `confidence: float`, `timestamp` | every `video_frame` |
| `transcript_chunk` | `text: str`, `timestamp` | every `audio_chunk` |
| `audio_debug` | `timings_ms` (ffmpeg_decode/whisper_cpp/total), `stt_error`, `byte_length` | every `audio_chunk` |
| `error` | `message: str` | on failure |

---

## Session Lifecycle

1. User selects a profile → `POST /api/v1/profiles/:id/select` → Express stores `profileId` in server-side session cookie.
2. Page loads → `+page.server.ts` calls `GET /api/v1/session`, `GET /api/v1/profiles`, `GET /api/v1/history` server-side.
3. Browser opens WebSocket to `ws://localhost:8000/ws` → sends `{type:"session_start", profile_id}`.
4. Model service creates `HarnessSession` in `_sessions[profile_id]` with a fresh `EmotionBuffer` and empty `transcript_buffer`.
5. Video frames stream every 200 ms → face detection + emotion model → `emotion_buffer` fills up.
6. Mic audio streams when speech detected → STT → `transcript_buffer` fills up.
7. User sends chat message → browser → SvelteKit → Express → `POST /api/v1/chat` on model service.
8. Chat handler reads `emotion_buffer.history()` + `transcript_buffer[-20:]` from session.
9. `EmotionalReasoningAgent.analyse()` produces emotional context string. Both it and the transcript segments are passed separately to `LLMReasoningAgent.reason()` → calls LLM.
10. Reply flows back: model service → Express (persists both turns) → SvelteKit → browser.
11. On page unload: WS sends `session_end` → `_sessions` entry deleted.

---

## What Is and Isn't Working

| Component | File | Status |
|---|---|---|
| Face detection (YOLOv8) | `core/face_detector.py` | Working |
| Emotion model (EmpathBotV1) | `core/emotion/empathbot.py` | Working — selected via `EMOTION_MODEL_ID=empathbot_final` |
| Emotion model (resnet18) | `core/emotion/resnet18.py` | Working — vanilla ResNet18, needs a compatible checkpoint |
| Emotion model (placeholder) | `core/emotion/placeholder.py` | Working — emits random labels (testing only) |
| Emotion buffer | `core/emotion/buffer.py` | Working |
| Debug flags | `core/debug_flags.py` | Working — runtime cycle / force / log overrides |
| STT (whisper.cpp) | `core/stt/whisper_cpp.py` | Working |
| STT (faster-whisper) | `core/stt/whisper_faster.py` | Implemented, tested less |
| Emotional reasoning | `core/emotional_reasoning_agent.py` | Basic — mode only, ignores confidence + transcript |
| LLM reasoning | `core/llm/reasoning_agent.py` | Working — static window and prompt |
| LLM (OpenAI) | `core/llm/openai.py` | Working |
| LLM (Gemini) | `core/llm/gemini.py` (if present) / via `LLM_PROVIDER=gemini` | Working |
| LLM (Anthropic) | `core/llm/anthropic.py` | **Stub — NotImplementedError** |
| LLM (Ollama) | `core/llm/ollama.py` | Implemented, untested |
| Profile persistence | `backend/src/lib/profileStore.ts` | Working |
| Conversation history | `backend/src/routes/history.router.ts` | Working |
| Transcript → reasoning | `ws/handler.py` + `emotional_reasoning_agent.py` | Collected but unused (low-hanging fruit) |
| Face cropper (CLI/library) | `face_cropper.py`, `face_cropper/` | Working — re-exports `FaceDetector` for notebooks |

To bypass the model during development, set `EMOTION_CYCLE_TEST_LABELS=true` (rotates through `EMOTIONS` on a timer) or `EMOTION_FORCE_LABEL=<label>` (pins one). Both can also be flipped at runtime via `core/debug_flags.py`.

---

## Stage History

| Stage | Folder | Question answered |
|---|---|---|
| 1 — Research | `sandbox/` | Which face detector? Which STT? |
| 2 — Integration | `application/mock_programs/` | Does the full pipeline work end-to-end? |
| 3 — Product | `application/` | Can a real user use it? |

`application/mock_programs/` is **deprecated** — do not use it as a reference or add to it. The production application (`application/`) supersedes it entirely.
