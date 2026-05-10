# Architecture Guide

A walkthrough of how the system is structured and how data flows through it — including exact function call chains, integration points for the emotion model, and where to expand LLM reasoning.

---

## The Big Picture

This system is an **emotion-aware study companion**. The core idea: read the student's face via webcam, classify their emotional state in real time, and condition the LLM's responses on that emotion — so a frustrated student gets more patient explanations, not faster ones.

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
│    └── Emotion model  ← PLACEHOLDER — needs real model      │
│    └── whisper.cpp STT                                      │
│    └── LLM (OpenAI / Ollama / Anthropic)                   │
└─────────────────────────────────────────────────────────────┘
```

**Key constraint**: the browser never calls Express or the model service directly. All HTTP calls go through SvelteKit server-side. The only exception is the WebSocket, which the browser opens directly to the model service at `ws://localhost:8000/ws`.

---

## Three Parallel Data Flows

There are three simultaneous flows that all converge on the LLM when the user sends a chat message.

### Flow 1 — Video → Face Detection → Emotion Buffer

This is the continuous stream that populates the emotional context used by the LLM.

```
frontend/src/routes/+page.svelte
  captureVideoFrame()             — canvas.drawImage(videoEl) → toDataURL("image/jpeg")
  ws.send({type:"video_frame", data: base64JPEG, timestamp})
  [every FRAME_INTERVAL_MS = 500 ms]
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
    │  ┌─────────────────────────────────────────────────────────────────┐
    │  │  ⚠ EMOTION MODEL INTEGRATION POINT (handler.py lines ~311–315) │
    │  │                                                                 │
    │  │  CURRENT (placeholder):                                         │
    │  │    if config.TEST_EMOTIONS:                                     │
    │  │        emotion = random.choice(EMOTIONS)   # always random      │
    │  │    else:                                                         │
    │  │        emotion = "happy" if detected else random.choice(EMOTIONS)│
    │  │    confidence = 0.95 if detected else random.uniform(0.5, 0.8)  │
    │  │                                                                 │
    │  │  REPLACE WITH:                                                  │
    │  │    emotion_model = getattr(app.state, "emotion_model", None)    │
    │  │    if detected and face_crop is not None and emotion_model:      │
    │  │        emotion, confidence = emotion_model.predict(face_crop)   │
    │  │        # predict() lives in core/emotion/base.py (ABC)          │
    │  │        # real implementation goes in core/emotion/<name>.py      │
    │  │        # register it in core/emotion/factory.py                 │
    │  │        # set EMOTION_VARIANT=<name> in .env                     │
    │  │    elif config.TEST_EMOTIONS:                                    │
    │  │        emotion = random.choice(EMOTIONS)                        │
    │  │        confidence = random.uniform(0.5, 0.8)                    │
    │  │    else:                                                         │
    │  │        emotion = "neutral"                                       │
    │  │        confidence = 0.5                                          │
    │  └─────────────────────────────────────────────────────────────────┘
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

This flow captures what the student says between keystrokes and stores it as transcript segments. The transcript buffer is passed to `EmotionalReasoningAgent` when a chat message is sent (currently not used in reasoning — see expansion point below).

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
- The transcript buffer is accessible at `session.transcript_buffer[-20:]` — passed to `EmotionalReasoningAgent.analyse()` but **not yet used** in reasoning (see expansion point below)

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
         returns: "The student appears to be feeling {dominant} (~{N}s).
                   Calibrate tone accordingly without referencing this directly."

    llm_agent.reason(body.message, ctx, history)
    └─ core/llm/reasoning_agent.py  LLMReasoningAgent.reason()
         messages = []
         messages.append({role:"system", content: SYSTEM_PROMPT})   # static persona
         prior = [m for m in history if m["role"] in ("user","assistant")]
         messages.extend(prior[-history_window:])                    # last 10 turns
         messages.append({role:"system", content: emotional_context}) # emotion injection
         messages.append({role:"user", content: message})            # current turn
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

### 1. Real Emotion Model (`⚠ highest priority`)

**Where:** `application/model_service/ws/video.py`, function `_pick_emotion()`.

**Current state:** ignores `face_crop` entirely and returns a random emotion label when `TEST_EMOTIONS=true` (default), or `"neutral"` otherwise.

**What to do:**

Step 1 — Implement `EmotionModel` in a new file:
```
application/model_service/core/emotion/<your_model_name>.py
```
Must implement the ABC from `core/emotion/base.py`:
```python
def predict(self, face_bgr: np.ndarray) -> tuple[str, float]:
    # face_bgr: uint8 BGR crop of the detected face, any size
    # returns: (emotion_label, confidence) where label ∈ EmotionModel.EMOTIONS
    #   EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','neutral']
```

Step 2 — Register it in the factory:
```python
# core/emotion/factory.py
if variant == "your_model_name":
    from .your_model_name import YourEmotionModel
    return YourEmotionModel()
```

Step 3 — Set the env var:
```
# application/model_service/.env
EMOTION_VARIANT=your_model_name
TEST_EMOTIONS=false
```

Step 4 — `_pick_emotion()` in `ws/video.py` already has the right structure. Just ensure `EMOTION_VARIANT` points to your new model and `TEST_EMOTIONS=false` in `.env` — the function will call `emotion_model.predict(face_crop)` automatically when the model is loaded.

The face crop is already in BGR uint8 format from the face detector — resize it to match your model's expected input inside `predict()`.

---

### 2. LLM Reasoning Expansion

There are three layers where reasoning can be expanded, ordered from easiest to most impactful:

#### Layer A — `EmotionalReasoningAgent.analyse()` in `core/emotional_reasoning_agent.py`

**Current:** uses `statistics.mode` over raw emotion labels, ignores confidence scores, ignores transcript segments entirely.

**Expand to:**
- Weight recent observations more heavily (recency-biased mode or exponential moving average)
- Factor in confidence: low-confidence observations should count less
- Use `transcript_segments` (currently passed in but unused): scan for verbal frustration cues ("I don't understand", "this doesn't make sense") to reinforce the emotional signal
- Produce richer context: include confidence, duration, trend (improving/worsening), and key transcript phrases

Example of a richer output:
```
"The student has appeared frustrated (angry/sad) for ~18s (confidence 0.82, worsening).
They recently said: 'I don't understand why this works'.
Slow down significantly. Offer a simpler analogy before re-explaining."
```

This string is injected verbatim as a `system` role message directly before the user's turn — so more specific language translates directly into more tailored LLM behaviour.

#### Layer B — `LLMReasoningAgent.reason()` in `core/llm/reasoning_agent.py`

**Current:** static 10-message history window; emotional context is a single sentence injected as a system message.

**Expand to:**
- Dynamically adjust `history_window` based on emotional state: frustrated students benefit from a longer window (more context on what's confused them); anxious students may benefit from a shorter, more focused window
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
  ├─ "video_frame" → ws/video.py: process_video_frame()
  │    ws/video.py: _run_face_detection()
  │      core/face_detector.py: detect_best()             [WORKING]
  │        └─ returns face_crop (BGR uint8)
  │    ws/video.py: _pick_emotion()                        [PLACEHOLDER → NEEDS REAL MODEL]
  │      core/emotion/*.py: emotion_model.predict(face_crop)
  │        └─ returns (emotion, confidence)
  │    core/emotion/buffer.py: EmotionBuffer.update()      [WORKING]
  │
  └─ "audio_chunk" → ws/audio.py: process_audio_chunk()
       ws/audio.py: decode_browser_audio_to_numpy()        [WORKING]
         └─ ffmpeg WebM → float32 PCM
       core/stt/whisper_cpp.py: stt.transcribe(audio_np)  [WORKING]
         └─ returns (text, lang, conf)
       session.transcript_buffer.append()                  [WORKING, unused in reasoning]

routers/chat.py: chat()
  │
  ├─ ws/session.py: get_session(profile_id)
  ├─ emotion_agent.analyse()                core/emotional_reasoning_agent.py  [BASIC → EXPAND]
  │    uses: emotion_buffer.history()
  │    ignores: transcript_buffer           ← LOW-HANGING FRUIT: use this
  │
  └─ llm_agent.reason()                    core/llm/reasoning_agent.py        [WORKING → EXPAND]
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
    ├── session.router.ts      GET / → returns {profileId} from session
    └── prediction.router.ts   Stub (returns static string)
```

**Profile storage:** each profile gets `data/profiles/<uuid>.json` → `{profile, messages[]}`. `data/profiles/index.json` is a flat list of profile metadata only. The index is read on every list/lookup call (no in-memory cache).

### Model Service (`application/model_service/`)

FastAPI + Python. All ML work happens here.

```
app.py           FastAPI app creation, lifespan (loads all ML components), WS + router mounting
config.py        All env vars (STT_ENGINE, EMOTION_VARIANT, LLM_PROVIDER, TEST_EMOTIONS…)
main.py          Uvicorn entry point (used for direct python main.py; make dev uses uvicorn directly)
routers/
├── chat.py      POST /api/v1/chat  — emotion-aware LLM reply
└── prediction.py  Stub
ws/
├── handler.py   Thin dispatcher — accepts WS connection, routes messages to audio/video handlers
├── session.py   HarnessSession dataclass, _sessions store, get_session, emit_debug
├── audio.py     decode_browser_audio_to_numpy (ffmpeg) + process_audio_chunk (STT pipeline)
├── video.py     encode_jpeg_b64, _run_face_detection, _pick_emotion, process_video_frame
└── protocol.py  Dataclass definitions for all WS message types (documentation reference)
core/
├── face_detector.py             YOLOv8 face detector (HuggingFace, auto-downloaded + cached)
├── emotional_reasoning_agent.py EmotionObservation[] + transcript[] → context string
├── emotion/
│   ├── base.py       EmotionModel ABC  — implement this to add a real model
│   ├── buffer.py     EmotionBuffer (deque, window=10) + EmotionObservation dataclass
│   ├── factory.py    create_emotion_model(variant) — add new variants here
│   └── placeholder.py  Returns random emotion; used until real model is integrated
├── llm/
│   ├── base.py           LLMProvider ABC + Message TypedDict
│   ├── openai.py         OpenAI Chat Completions (working)
│   ├── anthropic.py      Stub — raises NotImplementedError
│   ├── ollama.py         Ollama local LLM (implemented, untested)
│   ├── factory.py        create_llm(provider, model)
│   └── reasoning_agent.py  LLMReasoningAgent — assembles full message list per turn
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
| `video_frame` | `data: base64 JPEG`, `timestamp: float` | every 500 ms |
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
5. Video frames stream every 500 ms → face detection + (placeholder) emotion → `emotion_buffer` fills up.
6. Mic audio streams when speech detected → STT → `transcript_buffer` fills up.
7. User sends chat message → browser → SvelteKit → Express → `POST /api/v1/chat` on model service.
8. Chat handler calls `get_session(profile_id)` → reads `emotion_buffer.history()` + `transcript_buffer[-20:]`.
9. `EmotionalReasoningAgent.analyse()` produces context string → `LLMReasoningAgent.reason()` calls LLM.
10. Reply flows back: model service → Express (persists both turns) → SvelteKit → browser.
11. On page unload: WS sends `session_end` → `_sessions` entry deleted.

---

## What Is and Isn't Working

| Component | File | Status |
|---|---|---|
| Face detection (YOLOv8) | `core/face_detector.py` | Working |
| Emotion model | `core/emotion/placeholder.py` | **Placeholder — random output** |
| Emotion buffer | `core/emotion/buffer.py` | Working (feeds real model when ready) |
| STT (whisper.cpp) | `core/stt/whisper_cpp.py` | Working |
| STT (faster-whisper) | `core/stt/whisper_faster.py` | Implemented, tested less |
| Emotional reasoning | `core/emotional_reasoning_agent.py` | Basic — mode only, ignores transcript |
| LLM reasoning | `core/llm/reasoning_agent.py` | Working — static window and prompt |
| LLM (OpenAI) | `core/llm/openai.py` | Working |
| LLM (Anthropic) | `core/llm/anthropic.py` | **Stub — NotImplementedError** |
| LLM (Ollama) | `core/llm/ollama.py` | Implemented, untested |
| Profile persistence | `backend/src/lib/profileStore.ts` | Working |
| Conversation history | `backend/src/routes/history.router.ts` | Working |
| Transcript → reasoning | `ws/handler.py` + `emotional_reasoning_agent.py` | Collected but unused |

`TEST_EMOTIONS=true` (default) bypasses the emotion model entirely and emits random emotions — lets you develop and test the UI and LLM integration before the real model is ready.

---

## Stage History

| Stage | Folder | Question answered |
|---|---|---|
| 1 — Research | `sandbox/` | Which face detector? Which STT? |
| 2 — Integration | `application/mock_programs/` | Does the full pipeline work end-to-end? |
| 3 — Product | `application/` | Can a real user use it? |

`application/mock_programs/` is **deprecated** — do not use it as a reference or add to it. The production application (`application/`) supersedes it entirely.
