# COMPSCI-731 Human-Robot Interaction — Team Project

An **emotion-aware empathy bot**. A webcam reads the user's face, a trained classifier labels their emotional state, and an LLM adapts its response using two separate inputs: the emotional signal from the face and a timestamped transcript from speech-to-text.

The bot shifts tone in real time based on non-verbal cues: more patient when the user looks frustrated or sad, more calming when anxious. The same input can produce a different reply depending on how the person appears.

---

## Run it in 60 seconds

```bash
# First time only
make install

# Start all three services
make dev
```

Then open `http://localhost:5173`.

`make dev` brings up:
- **Frontend** (SvelteKit) → `localhost:5173`
- **Backend** (Express) → `localhost:3001`
- **Model service** (FastAPI) → `localhost:8000`

The browser talks to the frontend; everything else is internal.

If ports are busy: `make kill`. To clean up cleanly: Ctrl-C, then `make kill`.

---

## What's where

```
.
├── application/         The three-service web app (frontend + backend + model_service)
├── face_cropper/        CLI + library wrapping the production face detector
│                          for use in training notebooks
├── face_cropper.py      ↑ the actual entry point (re-exports the model service's
│                          FaceDetector — single source of truth)
├── training_pipeline/   ML training harness (YAML config, step persistence, resumption)
├── sandbox/             Per-student exploratory research
├── experiments/         Shared cross-team experiments
├── report/              Academic paper (Markdown → PDF via pandoc)
├── models/              Downloaded model weights (gitignored)
└── Makefile             dev / install / kill / crop-faces / report targets
```

For deeper internals — request flow, WebSocket protocol, file-by-file map, "what's working / what isn't" — see [ARCHITECTURE.md](ARCHITECTURE.md).
Reference for AI-pair-programming tools: [CLAUDE.md](CLAUDE.md).

---

## I'm new to this repo. Where do I start?

Pick the doc that matches what you're about to do — they're each focused and short.

| You want to… | Read this first |
|---|---|
| Run the app and see it work | This file (above) |
| Add or change a frontend component | [ARCHITECTURE.md](ARCHITECTURE.md) → "Service Internals" → Frontend, plus `application/frontend/src/.example/Svelte5Reference.svelte` |
| Change a backend route | [ARCHITECTURE.md](ARCHITECTURE.md) → "Service Internals" → Backend |
| Add a new emotion model | [ARCHITECTURE.md](ARCHITECTURE.md) → "Integration Points" |
| Train a model in a notebook | "Training a model" section below — uses the face cropper to preprocess data |
| Work on a feature without breaking other peoples' branches | [WORKTREES.md](WORKTREES.md) — parallel worktrees with isolated ports |
| Understand the team's branching philosophy | [CONTRIBUTIONS.md](CONTRIBUTIONS.md) |
| Write the report | [report/README.md](report/README.md) |

---

## Training a model

The emotion model is currently **EmpathBotV1** (EfficientNet-B2 backbone, EmpathBot 6-class schema). It loads from `models/empathbot/empath_final.pth` via `application/model_service/models.yaml`.

If you're training your own model — in Kaggle, Colab, or a local notebook — you need to crop faces from your raw dataset first so your classifier trains on the same inputs the live service will hand it at inference.

**Use the face cropper.** Same `FaceDetector` class the model service uses, no duplication:

```bash
# Bulk preprocess a dataset
make crop-faces INPUT=./raw_dataset OUTPUT=./crops RESIZE=224 PADDING=0.1

# Or directly
python face_cropper.py crop-dir ./raw ./crops --recursive --resize 224 --report report.json
```

In a notebook:

```python
from face_cropper import crop_face
face = crop_face("path/to/image.jpg")   # PIL.Image or None
```

Walk through [`face_cropper/demo.ipynb`](face_cropper/demo.ipynb) for a runnable demo, and see [`face_cropper/README.md`](face_cropper/README.md) for the three usage patterns + tips. The training pipeline harness itself lives in [`training_pipeline/`](training_pipeline/README.md) if you want step-level persistence and YAML config merging.

---

## Plugging a new trained model into the service

`application/model_service/models.yaml` is a registry — id → checkpoint path + model class. Pick a slot, add an entry, set the env var:

```yaml
# application/model_service/models.yaml
models:
  my_model_v3:
    path:    models/my_model_v3.pth     # under gitignored models/
    variant: empathbot                  # or "resnet18", or your own variant
```

```bash
# application/model_service/.env
EMOTION_MODEL_ID=my_model_v3
```

Restart `make dev` and the service picks it up. If you need a new architecture (different from `empathbot` / `resnet18`), see [ARCHITECTURE.md](ARCHITECTURE.md) → "Integration Points" — it's a four-step recipe.

**Debugging emotion behaviour live:** there are runtime flags in `core/debug_flags.py` for cycling through labels, pinning a single label, or logging every prediction. Set them in `.env` or flip them in code. CLAUDE.md has the full list.

---

## Environment variables

Each service has a `.env` (gitignored) and a committed `.env.example`. Copy `.env.example` → `.env` in each service the first time you set up.

| Service | File | Key variables |
|---|---|---|
| backend | `application/backend/.env` | `PORT`, `NODE_ENV`, `MODEL_SERVICE_URL` |
| frontend | `application/frontend/.env` | `PUBLIC_BACKEND_URL`, `PUBLIC_HARNESS_WS_URL` |
| model_service | `application/model_service/.env` | `PORT`, `HOST`, `LLM_PROVIDER`, `LLM_MODEL`, `EMOTION_MODEL_ID`, `OPENAI_API_KEY` / `GEMINI_API_KEY` |

Full env-var catalogue: [CLAUDE.md](CLAUDE.md).

---

## Code conventions

### Frontend (Svelte 5 runes — always)

```svelte
let x = $state(value)          // not: let x = value
let y = $derived(expr)         // not: $: y = expr
$effect(() => { ... })         // not: $: { ... }
let { prop } = $props()        // not: export let prop
onclick={fn}                   // not: on:click={fn}
```

- Components: `src/lib/components/`
- Import via `$lib` alias: `import Foo from "$lib/components/Foo.svelte"`
- Server-side logic (backend calls) goes in `+page.server.ts`
- The browser must never call Express or the model service directly
- See `src/.example/Svelte5Reference.svelte` for a local cheat sheet

### Backend (Express / TypeScript)

- `src/index.ts` — server startup only
- `src/app.ts` — app creation, middleware, route mounting
- `src/config/env.ts` — all environment variables in one place
- `src/routes/<domain>.router.ts` — one file per domain
- All API routes prefixed `/api/v1/`
- Every async route handler wraps in `try/catch` and calls `next(err)` on failure
- Run `npm run typecheck` before committing

### Model Service (FastAPI / Python)

- `main.py` — entry point only
- `app.py` — FastAPI app creation and router registration
- `config.py` — all environment variables + `load_model_registry()` for models.yaml
- `routers/<domain>.py` — one file per domain
- All routes prefixed `/api/v1/`
- Use Pydantic `BaseModel` for all request/response bodies
- Factory pattern for swappable ML backends: `base.py` (ABC) → `<name>.py` → `factory.py`

---

## Workflow

Branches and PRs follow the philosophy in [CONTRIBUTIONS.md](CONTRIBUTIONS.md). Short version:

- `main` is the shared trunk
- `sandbox/student_<name>/` is each person's exploratory space
- `application/`, `training_pipeline/`, `report/` are protected — changes go through a PR
- Branch names should describe a **question** being answered, not a ticket: `invest/<q>` for research, `integration/<q>` for landing the result, `feat/<thing>` for product work

For working on multiple branches in parallel without port collisions, use [WORKTREES.md](WORKTREES.md).

---

## History

This project moved through three stages. Each stage answered a question before the next began.

| Stage | Folder | Question |
|---|---|---|
| 1 — Research | `sandbox/` | Which face detector? Which STT backend? |
| 2 — Integration | `application/mock_programs/` (**deprecated**) | Does the pipeline work end-to-end? |
| 3 — Product | `application/` | Can a real user use it? |

`application/mock_programs/` is **deprecated** — do not extend it or use it as a reference. The production application supersedes it entirely.
