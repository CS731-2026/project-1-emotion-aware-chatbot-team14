# COMPSCI-731 Human-Robot Interaction — Team Project

## What we're building

An **emotion-aware study companion** that watches a student's face via webcam and adapts its conversational behaviour in real time.

Academic self-study is emotionally volatile — students cycle through frustration, helplessness, and anxiety, yet every AI study tool today responds the same way regardless of emotional state. This system treats the face as an honest, unfiltered signal of cognitive and emotional load. When the model detects frustration (anger, disgust, contempt, sadness) it shifts toward patient, scaffolded explanation. When it detects anxiety (fear) it moves to calming, confidence-building dialogue. The LLM prompt is dynamically conditioned on the detected emotion so that the *same question* receives a different answer depending on how the student looks.

The target population is university students in independent study — approximately 40,000 enrolled at the University of Auckland alone.

---

## The journey

This project moves through four stages. Each stage answered a question before the next one began.

```
Stage 1 — Research     What is the right approach? (sandbox/)
Stage 2 — Integration  Does it work end-to-end? (application/mock_programs/)
Stage 3 — Product      Can it be used by anyone? (application/)
```

---

## Repository structure

```
.
├── application/          The application code for our final product: frontend + backend + python ai model service
├── experiments/          any experiments we need to run (separate from our training pipeline)
├── sandbox/              A place for us to dump our files as we are working on our individual tasks, files that are pre mature for the application code (there will be a lot of these)
├── training_pipeline/    The training pipeline we are going to use for our hand trained models
├── report/               where we will be writing our final report in markdown
├── models/               Downloaded model weights (.gitignored TODO: we need to remove this)
└── Makefile              Root orchestration
```

---

## Stage 1 — Research: `sandbox/`

Before building anything, each team member used their sandbox folder to answer a specific question.

```
sandbox/
├── student_taurajgreig/     Face detection + speech recognition research
├── student_preeti/
└── student-kanishka/
```

**Questions answered:**

| Question | Where | Result |
|---|---|---|
| Which face detector is fastest and most reliable? | `sandbox/student_taurajgreig/` | YOLOv8-face outperforms RetinaFace, MediaPipe, and Haar cascades |
| Which speech-to-text backend fits our constraints? | `sandbox/student_taurajgreig/services/` | whisper.cpp gives the best speed/accuracy tradeoff without an API dependency |

The sandbox follows a strict rule: exploratory work stays in `sandbox/student_[name]/` until it has answered its question.

### Running sandbox work

Each sandbox has its own setup. See `sandbox/student_[name]/README.md` for instructions.

A convenience script activates the correct environment:

```bash
source sandbox-activate.sh
```

---

## `experiments/`

A shared space for work that doesn't belong to any one person and isn't ready for the application. If something needs to be run, tested, or documented collaboratively but has no obvious home elsewhere, it lives here.

```bash
source experiments-activate.sh
```

---

## Stage 2 — Integration: `application/mock_programs/`

With working face detection and a trained emotion model, the next question was: does the full pipeline work end-to-end?

`mock_programs/` is a complete, runnable terminal chatbot that integrates every component:

```
application/mock_programs/
├── main.py               Entry point (terminal UI, conversation loop)
├── face_detector.py      YOLOv8-face detection with drawing utilities
├── emotion_inferencer.py Inference wrapper + rolling-window emotion smoothing
├── chatbot.py            LLM integration (OpenAI) + 3-way model comparison
└── speech.py             FasterWhisper speech-to-text
```

### Running the mock chatbot

```bash
cd application/mock_programs
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY

# Full pipeline: webcam + voice + chatbot
python main.py --checkpoint path/to/emotion_model.pt --voice

# No webcam (fixed mock emotion for testing)
python main.py --no_webcam --mock_emotion happy

# Text only
python main.py --no_webcam --no_voice
```

**Terminal controls:**
- Type a message and press Enter to send
- Press Enter with empty input to record voice (if `--voice` enabled)
- Type `reset` to clear history, `quit` to exit

See [application/mock_programs/README.md](application/mock_programs/README.md) for full setup including dataset and model download instructions.

---

## Stage 3 — Product: `application/`

The mock program proved the concept. The production application is a three-service web app that packages the same pipeline for real users.

### Request flow

```
Browser
  └─▶ SvelteKit frontend       localhost:5173
        └─▶ Express backend    localhost:3000
              └─▶ FastAPI model service   localhost:8000
                    └─▶ Face detector + emotion classifier + LLM
```

The browser only ever talks to SvelteKit. SvelteKit's server-side load functions call Express. Express forwards to the model service. The model service runs inference and calls the LLM.

### Getting started

```bash
# Install all dependencies (first time only)
make web-install

# Start all three services
make web-dev
```

`make web-install` creates a Python virtual environment at `application/model_service/.venv`.

### Environment variables

Each service has a `.env` file (gitignored) and a committed `.env.example`:

| Service | File | Key variables |
|---|---|---|
| backend | `application/backend/.env` | `PORT`, `NODE_ENV`, `MODEL_SERVICE_URL` |
| frontend | `application/frontend/.env` | `BACKEND_URL` |
| model_service | `application/model_service/.env` | `PORT`, `HOST` |

Copy `.env.example` to `.env` in each service directory when setting up a new environment.

### Services

#### frontend — SvelteKit (TypeScript)

| Tool | Purpose |
|---|---|
| SvelteKit | Meta-framework (routing, SSR, build) |
| Svelte 5 | UI component framework |
| Vite | Dev server and bundler |

#### backend — Express (TypeScript)

| Tool | Purpose |
|---|---|
| Express 4 | HTTP server and routing |
| tsx | Run TypeScript directly in dev |
| dotenv | Environment variable loading |

#### model_service — FastAPI (Python)

| Tool | Purpose |
|---|---|
| FastAPI | HTTP server and routing |
| Uvicorn | ASGI server |
| Pydantic | Request/response validation |

---

## Support infrastructure

### `training_pipeline/`

A reusable ML harness used by `experiments/`. Handles config merging (multiple YAML files deep-merged left to right), step-level persistence to disk, and automatic run resumption after failures.

**Key concepts:**
- **Store** — the only channel through which steps pass data; serialised to disk after each step
- **Step** — a plain function `(store, config) → Success | Failure`
- **Routine** — ordered list of steps with a named `runs/` directory

See [training_pipeline/README.md](training_pipeline/README.md) for full documentation.

**Common commands:**

| Command | What it does |
|---|---|
| `make train` | Run the training pipeline |

### `report/`

The academic paper is written in Markdown and compiled to PDF via pandoc and LaTeX.

**Common commands:**

| Command | What it does |
|---|---|
| `make report` | Build PDF |
| `make report-docx` | Build DOCX |
| `make report-clean` | Remove build artefacts |
| `make report-deps` | Install dependencies |

See [report/README.md](report/README.md) for authoring instructions and citation syntax.

---

## Workflow

This project uses a **research-first workflow**. Every branch represents a question, not a task.

```
invest/question-name     → explore and answer the question
integration/result-name  → merge the answer into the main application
```

- `main` — shared branch for the main application
- `sandbox/student_[name]/` — each team member's exploratory space
- `application/`, `training_pipeline/`, `report/` — production folders; changes require a PR and approval

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for the full workflow.

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
- `config.py` — all environment variables
- `routers/<domain>.py` — one file per domain
- All routes prefixed `/api/v1/`
- Use Pydantic `BaseModel` for all request/response bodies
- Always work inside `model_service/.venv`; never install packages globally
