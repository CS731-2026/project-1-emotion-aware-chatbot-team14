# Architecture Guide

A walkthrough of how the system is structured and how data flows through it. Read this to get oriented before touching any code.

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
│    └── Emotion model                                        │
│    └── whisper.cpp STT                                      │
│    └── LLM (OpenAI / Ollama / Anthropic)                   │
└─────────────────────────────────────────────────────────────┘
```

**Key constraint**: the browser never calls Express or the model service directly. All HTTP calls go through SvelteKit's server-side load functions. The only exception is the WebSocket connection, which the browser opens directly to the model service.

---

## Two Parallel Data Flows

There are two simultaneous flows that both converge on the LLM:

### Flow 1 — Video + Emotion (WebSocket)

```
Browser webcam
  → captures JPEG frame every 500 ms
  → base64-encodes it
  → sends {type: "video_frame", data: "..."} over WebSocket

model_service ws/handler.py
  → decodes JPEG → numpy array
  → runs YOLOv8 face detector (detect_best)
  → crops face region
  → [emotion model predicts emotion]  ← placeholder currently returns random
  → updates EmotionBuffer with (emotion, confidence, timestamp)
  → sends back: face_detection, frame_debug, emotion_update messages

Browser receives emotion_update
  → updates UI background colour (EMOTION_COLOURS in types.ts)
```

### Flow 2 — Audio + Transcript (WebSocket + HTTP)

```
Browser microphone
  → BrowserVadController monitors RMS level every 100 ms
  → when level > SPEECH_THRESHOLD: starts MediaRecorder
  → when silence > 600 ms: stops, encodes clip as base64 WebM
  → sends {type: "audio_chunk", data: "..."} over WebSocket

model_service ws/handler.py
  → process_audio_chunk() (runs in asyncio worker thread)
  → ffmpeg converts WebM → 16 kHz mono PCM
  → whisper.cpp transcribes PCM → text
  → appends TranscriptSegment to session.transcript_buffer
  → sends back: transcript_chunk, audio_debug messages

Browser receives transcript_chunk → displays in TranscriptHistory
```

### Flow 3 — Chat (HTTP)

```
User types message → ChatInput component
  → api.sendChat(text) → POST /api/v1/chat to Express backend

Express backend (chat.router.ts)
  → reads windowed history from profileStore
  → POST /api/v1/chat to model_service (with profile_id + history)
  → persists user + agent messages to profileStore
  → returns response to frontend

model_service (routers/chat.py)
  → get_session(profile_id) → reads EmotionBuffer from WebSocket session
  → EmotionalReasoningAgent.analyse(emotion_observations, transcript_segments)
     → returns: "The student appears to be feeling frustrated (~12s)."
  → LLMReasoningAgent.reason(message, emotional_context, history)
     → assembles: [system prompt] + [windowed history] + [emotion context] + [user message]
     → calls LLM → returns assistant reply
```

The key bridge: `get_session()` links the HTTP chat request to the WebSocket session's emotion buffer using `profile_id` as the key. The WebSocket and HTTP flows are decoupled but share state through the in-memory `_sessions` dict.

---

## Service Internals

### Frontend (`application/frontend/`)

Built with SvelteKit + Svelte 5. One page: `src/routes/+page.svelte`.

**Svelte 5 runes** are used throughout — `$state`, `$derived`, `$effect`, `$props()`. No legacy `export let` or `on:*` event syntax.

```
src/
├── routes/
│   ├── +page.svelte       Main UI (chat, webcam, debug panel)
│   └── +page.server.ts    Server-side load (fetches session + profiles + history)
├── lib/
│   ├── api.ts             Typed fetch wrappers for all backend endpoints
│   ├── harness/
│   │   ├── browserVad.ts  VAD controller class (mic → WS audio chunks)
│   │   └── types.ts       Constants + emotion/type definitions
│   └── components/
│       ├── ChatHistory.svelte
│       ├── ChatInput.svelte
│       ├── WebcamPreview.svelte
│       ├── DebugDashboard.svelte
│       ├── TranscriptHistory.svelte
│       ├── ProfileModal.svelte
│       ├── SpeakingCircle.svelte
│       ├── SideNotes.svelte
│       └── Button.svelte
```

The page sends webcam frames over WebSocket and chat messages via `api.sendChat()`. The `DebugDashboard` component renders timing data from `frame_debug` and `audio_debug` WS messages — useful for verifying the pipeline is working.

### Backend (`application/backend/`)

Express 4 + TypeScript. Thin proxy between the browser and the model service. Owns user sessions and message history.

```
src/
├── index.ts               Starts server, handles EADDRINUSE
├── app.ts                 CORS, express-session, route mounting
├── config/env.ts          All env vars
├── lib/profileStore.ts    File-based persistence (data/profiles/)
├── middleware/errorHandler.ts
└── routes/
    ├── index.ts           Mounts all routers under /api/v1/
    ├── chat.router.ts     Proxies to model service; fallback on failure
    ├── profiles.router.ts CRUD + session selection
    ├── history.router.ts  Read/write conversation history
    ├── session.router.ts  Returns current profileId from session
    └── prediction.router.ts  Stub
```

**Profile storage**: each profile gets a `data/profiles/<uuid>.json` file containing `{profile, messages[]}`. `data/profiles/index.json` is a flat list of profile metadata (no messages). This keeps the index small.

**Session**: `express-session` stores `profileId` in a server-side session. All chat and history routes require an active session (401 otherwise).

### Model Service (`application/model_service/`)

FastAPI + Python. Does all the ML work.

```
app.py                   FastAPI app + lifespan loader
config.py                All env vars
main.py                  Uvicorn entry point
routers/
├── chat.py              POST /api/v1/chat → emotion-aware LLM reply
└── prediction.py        Stub
ws/
├── handler.py           WebSocket multiplexer (session, video, audio)
└── protocol.py          Message type dataclasses
core/
├── face_detector.py     YOLOv8 (HuggingFace auto-download)
├── emotional_reasoning_agent.py   Converts emotion buffer → context string
├── emotion/
│   ├── base.py          EmotionModel ABC
│   ├── buffer.py        EmotionBuffer (rolling window) + EmotionObservation
│   ├── factory.py       create_emotion_model(variant)
│   └── placeholder.py   Random emotion stub
├── llm/
│   ├── base.py          LLMProvider ABC + Message TypedDict
│   ├── openai.py        OpenAI Chat Completions
│   ├── anthropic.py     Stub (not yet implemented)
│   ├── ollama.py        Ollama local LLM
│   ├── factory.py       create_llm(provider, model)
│   └── reasoning_agent.py  LLMReasoningAgent (assembles full prompt)
└── stt/
    ├── base.py          TranscriptionService ABC
    ├── whisper_cpp.py   whisper.cpp backend
    ├── whisper_faster.py faster-whisper backend
    └── factory.py       create_stt(engine, model)
```

**Lifespan pattern**: each ML component loads at startup inside `app.py`'s `lifespan()` function and is stored on `app.state`. Failure of one component doesn't block the others — the service degrades gracefully (e.g. face detector fails → no face crops, but emotion and LLM still work).

**Factory pattern**: every swappable backend (LLM, STT, emotion model) follows the same pattern:
- `base.py` — abstract class with the interface
- `<provider>.py` — concrete implementation
- `factory.py` — `create_X(variant, ...)` selects and instantiates the right class

---

## WebSocket Protocol

The browser opens a single WebSocket to `ws://localhost:8000/ws` and sends tagged JSON messages.

**Inbound (browser → server):**

| type | payload | when |
|---|---|---|
| `session_start` | `{profile_id}` | On page load or profile select |
| `video_frame` | `{data: base64 JPEG, timestamp}` | Every 500 ms |
| `audio_chunk` | `{data: base64 WebM, timestamp}` | After VAD detects speech |
| `session_end` | — | On page unload |

**Outbound (server → browser):**

| type | payload | triggered by |
|---|---|---|
| `connection_ack` | — | Initial connection |
| `message_ack` | `{message_type}` | Every inbound message |
| `harness_status` | component loaded flags | `session_start` |
| `face_detection` | `{detected, timestamp}` | Every `video_frame` |
| `frame_debug` | `{box, timings_ms, image_data, face_crop_data}` | Every `video_frame` |
| `emotion_update` | `{emotion, confidence, timestamp}` | Every `video_frame` |
| `transcript_chunk` | `{text, timestamp}` | Every `audio_chunk` |
| `audio_debug` | `{timings_ms, stt_error}` | Every `audio_chunk` |
| `error` | `{message}` | On failure |

---

## Session Lifecycle

1. **User selects a profile** → POST `/api/v1/profiles/:id/select` → Express stores `profileId` in server session.
2. **Page loads** → `+page.server.ts` fetches session + profiles + history server-side.
3. **Browser opens WebSocket** → sends `session_start` with `profile_id`.
4. **Model service creates `HarnessSession`** in `_sessions[profile_id]`.
5. **Video frames + audio chunks stream** continuously.
6. **User sends a chat message** → browser → SvelteKit → Express → model service `/api/v1/chat`.
7. **Chat handler reads `_sessions[profile_id].emotion_buffer`** to get emotional context.
8. **LLM reply** flows back up the chain → stored in profileStore → returned to browser.

---

## What Is and Isn't Working Yet

| Component | Status |
|---|---|
| Face detection (YOLOv8) | Working |
| Emotion model | **Placeholder** — returns random emotion |
| STT (whisper.cpp) | Working |
| LLM (OpenAI) | Working |
| LLM (Anthropic) | Stub — `NotImplementedError` |
| LLM (Ollama) | Implemented, untested |
| Profile persistence | Working |
| Conversation history | Working |

The `TEST_EMOTIONS=true` env var (default) makes the WebSocket handler emit random emotions regardless of face detection — this lets you test the UI and LLM context injection before the real emotion model is ready.

---

## Stage History

| Stage | Folder | Question answered |
|---|---|---|
| 1 — Research | `sandbox/` | Which face detector? Which STT? |
| 2 — Integration | `application/mock_programs/` | Does the full pipeline work end-to-end? |
| 3 — Product | `application/` | Can a real user use it? |

The mock programs (`application/mock_programs/`) are a terminal chatbot with the same pipeline. Useful reference for how the components were originally integrated before the web app was built.
