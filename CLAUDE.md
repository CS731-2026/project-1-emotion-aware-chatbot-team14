# CLAUDE.md — HRI Team Project Quick Reference

## What this project is

An **emotion-aware empathy bot**. A webcam reads the user's face, a trained emotion classifier labels their emotional state, and an LLM adapts its response using two separate inputs: (1) emotional context derived from the face signal, and (2) a timestamped transcript from speech-to-text. Three services run in parallel:

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
make crop-faces INPUT=<dir> OUTPUT=<dir>   # batch face-crop a dataset
```

For parallel branches with isolated ports, see "Working in parallel branches" in `README.md`.

## Top-level folders

| Folder | Purpose |
|---|---|
| `application/` | The production three-service web app |
| `face_cropper/` | CLI + library wrapping the production face detector for use in notebooks / dataset preprocessing |
| `sandbox/` | Per-student exploratory research (pre-application) |
| `experiments/` | Shared cross-team experiments |
| `training_pipeline/` | ML training harness (YAML config, step persistence) |
| `report/` | Academic paper in Markdown → PDF via pandoc |
| `models/` | Downloaded model weights (gitignored) |

`face_cropper.py` (repo root) is the CLI + library; `application/model_service/core/face_detector.py` is the canonical detector — `face_cropper` re-exports it, so notebooks and the live service share one implementation.

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
- `config.py` — all env vars + `load_model_registry()` helper
- `models.yaml` — local model registry (id → path + variant); selected at runtime via `EMOTION_MODEL_ID`
- `routers/chat.py` — POST `/api/v1/chat`; runs EmotionalReasoningAgent → LLMReasoningAgent
- `ws/handler.py` — WS dispatcher + `pick_emotion()` (the only place `emotion_model.predict()` is called)
- `ws/session.py` — `HarnessSession`, session store (`_sessions`), `get_session`, `emit_debug`
- `ws/audio.py` — ffmpeg decode + `process_audio_chunk` (STT pipeline)
- `ws/video.py` — frame decode, YOLO call, encode — prepares `face_crop` for the emotion model
- `ws/protocol.py` — dataclass definitions for all WS message types
- `core/__init__.py` — re-exports `debug_flags` for `from core import debug_flags`
- `core/debug_flags.py` — mutable runtime flags (cycle / force / log); seeded from `.env`, can be overridden in code
- `core/face_detector.py` — YOLOv8 face detector (HuggingFace, auto-downloaded)
- `core/emotion/base.py` — `EmotionModel` ABC + `EMOTIONS` list (EmpathBot 6-class)
- `core/emotion/buffer.py` — `EmotionBuffer` rolling-window smoother + `EmotionObservation`
- `core/emotion/placeholder.py` — random-emotion stub
- `core/emotion/resnet18.py` — vanilla ResNet18 variant
- `core/emotion/empathbot.py` — EmpathBotV1 (EfficientNet-B2 / ResNet18 with SE) ported from `Notebooks/6b_empathbot_v1_improvements.ipynb`
- `core/emotion/factory.py` — `create_emotion_model()`; checks `EMOTION_MODEL_ID` first, falls back to `EMOTION_VARIANT`
- `core/llm/base.py` — `LLMProvider` ABC + `Message` TypedDict
- `core/llm/openai.py` — OpenAI Chat Completions provider
- `core/llm/anthropic.py` — Anthropic stub (not yet implemented)
- `core/llm/ollama.py` — Ollama local provider
- `core/llm/factory.py` — `create_llm(provider, model)`
- `core/llm/reasoning_agent.py` — `LLMReasoningAgent`; assembles system prompt + history + emotional context + transcript
- `core/emotional_reasoning_agent.py` — `EmotionalReasoningAgent`; produces emotion context string from buffer
- `core/stt/base.py` — `TranscriptionService` ABC
- `core/stt/whisper_cpp.py` — whisper.cpp backend (default)
- `core/stt/whisper_faster.py` — faster-whisper backend
- `core/stt/factory.py` — `create_stt(engine, model)`

### `application/mock_programs/` — **DEPRECATED**
Do not use or extend. Superseded by the production application.

## Emotion model selection

Two ways to pick which model loads — `EMOTION_MODEL_ID` wins if both are set.

**Preferred — registry:**
```
EMOTION_MODEL_ID=empathbot_final
```
`models.yaml` resolves the id to a path under `models/` (gitignored) and the variant class to instantiate.

**Legacy fallback — direct:**
```
EMOTION_VARIANT=empathbot
EMOTION_CHECKPOINT_PATH=models/empathbot/empath_final.pth
```

## Debug flags (`core/debug_flags.py`)

Mutable runtime overrides for the emotion pipeline. Defaults come from `.env`; any code (e.g. `routers/chat.py`) can flip them at runtime:

```python
from core import debug_flags
debug_flags.emotion.force_label = "sadness"        # pin a label
debug_flags.emotion.log_predictions = True
```

Resolution order in `ws/handler.py::pick_emotion()`:
1. `force_label` → pin a specific emotion (bypasses everything)
2. `cycle_test_labels` → step through `EMOTIONS` on a timer (bypasses model)
3. Real model prediction
4. Neutral fallback (no face / no model)

## Key env vars (model_service)

| Var | Default | Options / Notes |
|---|---|---|
| `STT_ENGINE` | `whisper-cpp` | `whisper-cpp`, `faster-whisper` |
| `EMOTION_MODEL_ID` | unset | An id in `models.yaml` (e.g. `empathbot_final`). Preferred over `EMOTION_VARIANT`. |
| `EMOTION_VARIANT` | `placeholder` | `placeholder`, `resnet18`, `empathbot`. Used only when `EMOTION_MODEL_ID` is unset. |
| `EMOTION_CHECKPOINT_PATH` | `models/resnet18_emotion.pth` | Only used when `EMOTION_MODEL_ID` is unset. |
| `EMOTION_DEVICE` | unset (auto) | `cpu`, `mps`, `cuda`. Auto-detected when unset. |
| `EMOTION_CYCLE_TEST_LABELS` | `false` | Debug: cycle through `EMOTIONS` on a timer instead of running the model. |
| `EMOTION_CYCLE_INTERVAL_SECONDS` | `30` | Seconds per label when cycling. |
| `EMOTION_FORCE_LABEL` | unset | Debug: pin a label (must be in `EMOTIONS`). |
| `EMOTION_LOG_PREDICTIONS` | `false` | Debug: log every prediction at INFO. |
| `LLM_PROVIDER` | `openai` | `openai`, `gemini`, `anthropic`, `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | any model string for the chosen provider |

## Face cropper for teammates

`face_cropper.py` (repo root) + `face_cropper/` (docs, demo, smoke test) — wraps the **same** `FaceDetector` class the model service uses, so dataset preprocessing in notebooks stays consistent with live inference.

- CLI: `python face_cropper.py crop-dir <in> <out> --recursive --resize 224 --padding 0.1 --report r.json`
- Library: `from face_cropper import crop_face` (accepts path / PIL / numpy)
- Demo: `face_cropper/demo.ipynb` (runnable end-to-end)

## Code conventions

- Python: factory pattern for all ML backends (base ABC → concrete → factory fn)
- TypeScript: one router file per domain; async handlers always `try/catch → next(err)`
- All API routes prefixed `/api/v1/`
- Browser never calls backend directly — all requests go via SvelteKit server-side
- `data/profiles/` holds per-profile JSON files + an `index.json`; gitignored in prod
