# CLAUDE.md — HRI Team Project Quick Reference

## What this project is

An emotion-aware study companion. A webcam reads the student's face, a trained emotion classifier labels their emotional state, and an LLM adapts its tone accordingly. Three services run in parallel:

```
Browser → SvelteKit (5173) → Express backend (3001) → FastAPI model service (8000)
                                                              ↕
                                               WebSocket (video frames + audio)
```

## Running the project

```bash
make dev          # start all three services (kills ports first)
make install      # npm install + pip install (first time only)
make kill         # free ports 3000/3001/5173/8000
```

## Top-level folders

| Folder | Purpose |
|---|---|
| `application/` | The production three-service web app |
| `sandbox/` | Per-student exploratory research (pre-application) |
| `experiments/` | Shared cross-team experiments |
| `training_pipeline/` | ML training harness (YAML config, step persistence) |
| `report/` | Academic paper in Markdown → PDF via pandoc |
| `models/` | Downloaded model weights (gitignored) |

## application/ — service layout

### `application/frontend/` — SvelteKit + Svelte 5
- `src/routes/+page.svelte` — single page (chat UI + webcam + debug panel)
- `src/routes/+page.server.ts` — server-side load (profile/session hydration)
- `src/lib/api.ts` — typed fetch wrappers for all backend endpoints
- `src/lib/harness/browserVad.ts` — browser-side VAD; detects speech, sends WebM audio over WS
- `src/lib/harness/types.ts` — shared constants (thresholds, frame interval, emotion map)
- `src/lib/components/` — UI components (ChatHistory, ChatInput, WebcamPreview, DebugDashboard, …)

**Svelte 5 runes only.** `$state`, `$derived`, `$effect`, `$props()`. No `export let`, no `on:*`.

### `application/backend/` — Express + TypeScript
- `src/index.ts` — server startup only
- `src/app.ts` — CORS, session middleware, route mounting
- `src/config/env.ts` — all env vars in one place
- `src/lib/profileStore.ts` — file-based profile + message store (`data/profiles/`)
- `src/routes/chat.router.ts` — forwards chat to model service; falls back gracefully
- `src/routes/profiles.router.ts` — CRUD for profiles; POST `/:id/select` sets session
- `src/routes/history.router.ts` — GET/POST conversation history for active profile
- `src/routes/session.router.ts` — GET current session profileId
- `src/routes/prediction.router.ts` — stub for future prediction endpoint

All routes under `/api/v1/`. All async handlers use `try/catch → next(err)`.

### `application/model_service/` — FastAPI + Python
- `main.py` — uvicorn entry point
- `app.py` — FastAPI app creation, lifespan (loads ML components), WS + router mounting
- `config.py` — all env vars (`STT_ENGINE`, `EMOTION_VARIANT`, `LLM_PROVIDER`, etc.)
- `routers/chat.py` — POST `/api/v1/chat`; runs EmotionalReasoningAgent → LLMReasoningAgent
- `routers/prediction.py` — stub
- `ws/handler.py` — thin WS dispatcher; routes messages to audio/video handlers
- `ws/session.py` — `HarnessSession`, session store (`_sessions`), `get_session`, `emit_debug`
- `ws/audio.py` — ffmpeg decode + `process_audio_chunk` (STT pipeline)
- `ws/video.py` — JPEG encode, YOLO call, `_pick_emotion`, `process_video_frame`
- `ws/protocol.py` — dataclass definitions for all WS message types
- `core/face_detector.py` — YOLOv8 face detector (HuggingFace, auto-downloaded)
- `core/emotion/base.py` — `EmotionModel` ABC
- `core/emotion/buffer.py` — `EmotionBuffer` rolling-window smoother + `EmotionObservation`
- `core/emotion/placeholder.py` — random emotion stub (default until real model integrated)
- `core/emotion/factory.py` — `create_emotion_model(variant)`
- `core/llm/base.py` — `LLMProvider` ABC + `Message` TypedDict
- `core/llm/openai.py` — OpenAI Chat Completions provider
- `core/llm/anthropic.py` — Anthropic stub (not yet implemented)
- `core/llm/ollama.py` — Ollama local provider
- `core/llm/factory.py` — `create_llm(provider, model)`
- `core/llm/reasoning_agent.py` — `LLMReasoningAgent`; assembles system prompt + history + emotion context
- `core/emotional_reasoning_agent.py` — `EmotionalReasoningAgent`; produces emotion context string from buffer
- `core/stt/base.py` — `TranscriptionService` ABC
- `core/stt/whisper_cpp.py` — whisper.cpp backend (default)
- `core/stt/whisper_faster.py` — faster-whisper backend
- `core/stt/factory.py` — `create_stt(engine, model)`

### `application/mock_programs/` — **DEPRECATED**
Do not use or extend. Superseded by the production application.

## Key env vars (model_service)

| Var | Default | Options |
|---|---|---|
| `STT_ENGINE` | `whisper-cpp` | `whisper-cpp`, `faster-whisper` |
| `EMOTION_VARIANT` | `placeholder` | `placeholder` (real model TBD) |
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | any model string for the chosen provider |
| `TEST_EMOTIONS` | `true` | set `false` to use face detector output |

## Code conventions

- Python: factory pattern for all ML backends (base ABC → concrete → factory fn)
- TypeScript: one router file per domain; async handlers always `try/catch → next(err)`
- All API routes prefixed `/api/v1/`
- Browser never calls backend directly — all requests go via SvelteKit server-side
- `data/profiles/` holds per-profile JSON files + an `index.json`; gitignored in prod
