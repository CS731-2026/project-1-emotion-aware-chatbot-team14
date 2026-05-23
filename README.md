# COMPSCI-731 Human-Robot Interaction — Team Project

An **emotion-aware empathy bot**. A webcam reads the user's face, a trained classifier labels their emotional state, and an LLM adapts its response using two separate inputs: the emotional signal from the face and a timestamped transcript from speech-to-text.

The bot shifts tone in real time based on non-verbal cues: more patient when the user looks frustrated or sad, more calming when anxious. The same input can produce a different reply depending on how the person appears.

---

## Mental model — what this repo is

Three layers stacked on the same git repo:

```
┌──────────────────────────────────────────────────────────────────────┐
│  application/        the live web app (3 services + a secret test page) │
│      ↑ loads the trained model from models.yaml at boot                │
├──────────────────────────────────────────────────────────────────────┤
│  pipeline/           the training pipeline (datasets, models, configs) │
│      ↑ produces the .pth checkpoints the app loads                     │
├──────────────────────────────────────────────────────────────────────┤
│  Notebooks/          original research notebooks (one per team member) │
│      ↑ source of truth for the model architectures ported into pipeline/ │
└──────────────────────────────────────────────────────────────────────┘
```

Three jobs the repo lets you do, in order of frequency:

| Job | What | Where to start |
|---|---|---|
| Run the app | `make dev` brings up SvelteKit + Express + FastAPI; opens at `localhost:5173` | "Run it in 60 seconds" below |
| Train a model | `make new-model ID=foo` → edit → `make train RUN=foo` → `make compare` | [pipeline/MIGRATING_NOTEBOOKS.md](pipeline/MIGRATING_NOTEBOOKS.md) |
| Ship a trained model | `make deploy-model RUN=LATEST ID=foo` → `EMOTION_MODEL_ID=foo` in `.env` → restart | [pipeline/MIGRATING_NOTEBOOKS.md § 7](pipeline/MIGRATING_NOTEBOOKS.md) |

The training pipeline is set up so a teammate only needs their Kaggle creds in `.env` — the live model service auto-fetches missing checkpoints from the team's Kaggle weights dataset on first boot.

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

**Speech-to-text needs one of two backends.** The default `STT_ENGINE=whisper-cpp` requires you to build [whisper.cpp](https://github.com/ggml-org/whisper.cpp) yourself — fast but ~5 minutes of setup. If you just want the app running, set `STT_ENGINE=faster-whisper` in `application/model_service/.env` instead. It's installed automatically by `make install` and works out of the box, just slower per request.

If neither STT backend loads, the model service still starts and the chat works — the mic just won't transcribe.

### Out of the box, what does it do?

With the default `.env`, the model service runs in **placeholder mode** — it emits random emotion labels instead of running a trained classifier. The UI, LLM calls, STT, and WebSocket pipeline are fully functional; only the emotion classifier is stubbed. This lets you exercise the whole app without needing any model files on disk.

### Switching to the real model

The real emotion model is **EmpathBotV1** (EfficientNet-B2, EmpathBot 6-class). The checkpoint (`empath_final.pth`) lives at `models/empathbot/empath_final.pth` under the gitignored `models/` directory.

1. **Get the file.** It's not in git (too large) and not yet hosted publicly. Ask another teammate to share it (Google Drive / Slack / etc.). A Kaggle dataset for this is planned but not live yet.
2. **Place it.** `models/empathbot/empath_final.pth` — create the directory if needed.
3. **Switch the service to use it.** In `application/model_service/.env`, uncomment:
   ```
   EMOTION_MODEL_ID=empathbot_final
   ```
4. Restart `make dev`. You should see `[model_service] INFO EmpathBot emotion model loaded: …` in the logs.

---

## What's where

```
.
├── application/         The three-service web app (frontend + backend + model_service)
│   └── frontend/src/routes/emotion-test/   bare webcam → emotion test page
│                                           (no chat, no LLM — model debugging)
├── pipeline/            Training pipeline. Drop in a model, declare a run in runs.yaml,
│                          make train. See pipeline/MIGRATING_NOTEBOOKS.md.
│   ├── models/tutorial/        heavily-commented reference model (read this first)
│   ├── datasets/tutorial/      same on the dataset side
│   ├── framework/              orchestration internals (you rarely touch)
│   └── training/               shared helpers (loops, losses, optimizers, reporting)
├── runs.yaml            What `make train` runs — one YAML entry per (dataset, model, config)
├── configs/             Hyperparameter presets (fast / baseline / thorough)
├── Notebooks/           Original research notebooks — source of truth for ports under pipeline/
├── face_cropper.py      CLI + library wrapping the production face detector
│                          (re-exports the model service's FaceDetector — single source of truth)
├── face_cropper/        Docs + demo + smoke test for the above
├── sandbox/             Per-student exploratory research
├── report/              Academic paper (Markdown → PDF via pandoc)
├── vendor/              git-weave .thread files for vendored third-party repos
│                          (POSTER_V2 cloned here at install time)
├── models/              Trained checkpoints (gitignored — auto-fetched from Kaggle)
├── output/              Run artifacts (gitignored — checkpoints, plots, metrics per run)
└── Makefile             dev / install / kill / train / deploy-model / publish-models / compare / new-model
```

For deeper internals — request flow, WebSocket protocol, file-by-file map, "what's working / what isn't" — see [ARCHITECTURE.md](ARCHITECTURE.md).
For a condensed file-by-file map and the full env-var catalogue: [CLAUDE.md](CLAUDE.md). (Originally written as instructions for AI pair-programming tools, but it's also the fastest way for a human to find which file owns which behaviour.)

---

## I'm new to this repo. Where do I start?

Pick the doc that matches what you're about to do — they're each focused and short.

| You want to… | Read this first |
|---|---|
| Run the app and see it work | This file (above) |
| Add or change a frontend component | [ARCHITECTURE.md](ARCHITECTURE.md) → "Service Internals" → Frontend, plus `application/frontend/src/.example/Svelte5Reference.svelte` |
| Change a backend route | [ARCHITECTURE.md](ARCHITECTURE.md) → "Service Internals" → Backend |
| **Train a new model** | [pipeline/MIGRATING_NOTEBOOKS.md](pipeline/MIGRATING_NOTEBOOKS.md) — the only doc you need. Read `pipeline/models/tutorial/` then `make new-model ID=your_name` |
| **Tweak hyperparameters for an existing model** | Edit the model's `CFG` in `pipeline/models/<name>/train_loop.py`, OR add a `train_cfg:` block to its row in `runs.yaml` |
| **Ship a trained model to the live app** | `make deploy-model RUN=LATEST ID=foo` then `EMOTION_MODEL_ID=foo` in `.env` |
| Test the emotion model without the chat surface | Open `http://localhost:5173/emotion-test/` after `make dev` |
| Work on a feature without breaking other peoples' branches | "Working in parallel branches" section below |
| Understand the team's branching philosophy | [CONTRIBUTIONS.md](CONTRIBUTIONS.md) |
| Write the report | [report/README.md](report/README.md) |

---

## Training a model

The team-wide training pipeline lives under `pipeline/`. Three commands cover the whole iteration loop:

```bash
make install-training              # one time — pip deps + git-weave + POSTER_V2 stage
make new-model ID=my_model         # scaffolds pipeline/models/my_model/ from the tutorial template
                                   #   (edit model.py + train_loop.py CFG, then:)
make train RUN=my_model            # picks just the matching row from runs.yaml
make compare FILTER=my_model       # leaderboard of every run you've done so far
```

10 of 18 declared runs in `runs.yaml` work without any extra setup (synthetic data; no Kaggle). The 8 FER2013 runs need a Kaggle API key in `.env`:

```env
# .env at repo root
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

**The full walkthrough is in [pipeline/MIGRATING_NOTEBOOKS.md](pipeline/MIGRATING_NOTEBOOKS.md)** — read that first if you're going to spend more than 10 minutes on training. It covers: the three-function mental model, how runs.yaml resolves names to function calls, where hyperparameters live (3 layers), what artifacts each run produces, how to deploy, how to share weights via Kaggle, and how to add a model from scratch.

**Reference modules to read top-to-bottom:**
- `pipeline/models/tutorial/` — every framework feature with inline comments
- `pipeline/datasets/tutorial/` — same on the dataset side

The face cropper is still around for raw-dataset preprocessing in notebooks: `python face_cropper.py crop-dir ./raw ./crops --recursive --resize 224` or `from face_cropper import crop_face`. Same `FaceDetector` class the live service uses.

> The legacy `training_pipeline/` directory is **archived** under `.archive/` and superseded entirely by `pipeline/`.

---

## Plugging a new trained model into the service

The training pipeline does this for you — `make deploy-model RUN=LATEST ID=foo` copies the checkpoint into `models/foo/best.pth` and writes the right entry in `application/model_service/models.yaml`. Then set `EMOTION_MODEL_ID=foo` in `.env` and restart `make dev`.

To pull a teammate's published model:

```env
# .env — Kaggle creds + the model id you want
KAGGLE_USERNAME=...
KAGGLE_KEY=...
EMOTION_MODEL_ID=their_model_id
```

`make dev` — the model service auto-fetches from the team's Kaggle weights dataset on first boot. Cached locally afterwards.

Manual yaml edit (if you have a `.pth` from somewhere else):

```yaml
# application/model_service/models.yaml
models:
  my_model_v3:
    path:    models/my_model_v3.pth     # under gitignored models/
    variant: empathbot                  # or "resnet18", or your own variant
```

If you need a new architecture (different from `empathbot` / `resnet18`), see [ARCHITECTURE.md](ARCHITECTURE.md) → "Integration Points" — it's a four-step recipe.

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
- `application/`, `pipeline/`, `report/` are protected — changes go through a PR
- Branch names should describe a **question** being answered, not a ticket: `invest/<q>` for research, `integration/<q>` for landing the result, `feat/<thing>` for product work

## Working in parallel branches

Use git worktrees to develop several branches simultaneously without port collisions or duplicating the multi-GB `models/` and `dataset/` directories.

```bash
# Create a new worktree branched off your current HEAD
git worktree add ../hri-worktrees/feat--my-thing -b feat/my-thing

# Inside the new worktree: set per-service ports so two `make dev` runs
# don't fight over 3001 / 5173 / 8000
cd ../hri-worktrees/feat--my-thing
echo "PORT=3011" >> application/backend/.env
echo "PUBLIC_BACKEND_URL=http://localhost:3011" > application/frontend/.env
# …and so on for model_service. See application/*/.env.example for the keys.
```

To save the heavy `models/` weights from being duplicated, symlink it from the main checkout:
```bash
ln -s /path/to/main-checkout/models ./models
```

The team also has helper scripts (`scripts/worktree-add.sh`, `worktree-list.sh`, `worktree-remove.sh`) that automate this — currently only on your local machine. Ask the maintainer to share them or copy them across; they're not committed.

---

## History

This project moved through three stages. Each stage answered a question before the next began.

| Stage | Folder | Question |
|---|---|---|
| 1 — Research | `sandbox/` | Which face detector? Which STT backend? |
| 2 — Integration | `application/mock_programs/` (**deprecated**) | Does the pipeline work end-to-end? |
| 3 — Product | `application/` | Can a real user use it? |

`application/mock_programs/` is **deprecated** — do not extend it or use it as a reference. The production application supersedes it entirely.
