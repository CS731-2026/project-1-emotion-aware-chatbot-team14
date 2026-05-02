# COMPSCI-731 Human-Robot Interaction — Team Project

## Repository Structure

```
.
├── report/               Academic paper (Markdown → LaTeX → PDF)
├── training_pipeline/    ML experimentation harness and model training
├── web_application/      Full-stack application (frontend, backend, model service)
└── Makefile              Root orchestration — delegates to each sub-project
```

---

## report/

The report is written in Markdown and compiled to a formatted PDF via pandoc and LaTeX. A separate DOCX output is also available.

See [report/README.md](report/README.md) for full authoring instructions, citation syntax, and template switching.

**Common commands (run from repo root):**

| Command | What it does |
|---|---|
| `make report` | Build PDF |
| `make report-docx` | Build DOCX |
| `make report-clean` | Remove build artefacts |
| `make report-deps` | Install dependencies |

---

## training_pipeline/

A modular ML experimentation harness built on PyTorch. Handles CLI argument parsing, config merging (multiple YAML files deep-merged left to right), per-step persistence to disk, and automatic run resumption.

See [training_pipeline/README.md](training_pipeline/README.md) for full documentation including Store, Steps, Routines, serialisation, and the working example pipeline.

**Key concepts:**

- **Store** — the only channel through which steps pass data; serialised to disk after each step
- **Step** — a plain function `(store, config) -> Success | Failure`
- **Routine** — ordered list of steps with a named `runs/` directory

---

## web_application/

A three-service web application. The frontend calls the backend, which forwards requests to the Python model service.

### Request flow

```
Browser
  └─▶ SvelteKit (SSR server)         localhost:5173
        └─▶ Express backend           localhost:3000
              └─▶ FastAPI model service   localhost:8000
                    └─▶ ML model (future)
```

All three services run independently. The browser only ever talks to SvelteKit. SvelteKit's server-side load functions call Express. Express forwards to the model service.

### Services

#### frontend — SvelteKit (TypeScript)

| Tool | Purpose | Docs |
|---|---|---|
| [SvelteKit](https://kit.svelte.dev) | Meta-framework (routing, SSR, build) | kit.svelte.dev |
| [Svelte 5](https://svelte.dev) | UI component framework | svelte.dev |
| [Vite 8](https://vite.dev) | Dev server and bundler | vite.dev |
| [TypeScript](https://www.typescriptlang.org) | Type checking | typescriptlang.org |
| [svelte-check](https://github.com/sveltejs/language-tools) | Svelte-aware type checking | github.com/sveltejs/language-tools |

#### backend — Express (TypeScript)

| Tool | Purpose | Docs |
|---|---|---|
| [Express 4](https://expressjs.com) | HTTP server and routing | expressjs.com |
| [TypeScript](https://www.typescriptlang.org) | Type checking | typescriptlang.org |
| [tsx](https://github.com/privatenumber/tsx) | Run TypeScript directly (dev) | github.com/privatenumber/tsx |
| [dotenv](https://github.com/motdotla/dotenv) | Environment variable loading | github.com/motdotla/dotenv |

#### model_service — FastAPI (Python)

| Tool | Purpose | Docs |
|---|---|---|
| [FastAPI](https://fastapi.tiangolo.com) | HTTP server and routing | fastapi.tiangolo.com |
| [Uvicorn](https://www.uvicorn.org) | ASGI server | uvicorn.org |
| [Pydantic](https://docs.pydantic.dev) | Request/response validation (bundled with FastAPI) | docs.pydantic.dev |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | Environment variable loading | github.com/theskumar/python-dotenv |

### Getting started

```bash
# Install all dependencies (first time only)
make web-install

# Start all three services
make web-dev
```

`make web-install` creates a Python virtual environment at `model_service/.venv` — Python packages are isolated from your system environment.

### Environment variables

Each service has a `.env` file (gitignored) and a committed `.env.example`:

| Service | File | Key variables |
|---|---|---|
| backend | `backend/.env` | `PORT`, `NODE_ENV`, `MODEL_SERVICE_URL` |
| frontend | `frontend/.env` | `BACKEND_URL` |
| model_service | `model_service/.env` | `PORT`, `HOST` |

Copy `.env.example` to `.env` when setting up a new environment.

---

## Conventions

### Frontend (SvelteKit / Svelte 5)

**Svelte syntax — always use Svelte 5 runes:**
- State: `let x = $state(value)` — never `let x = value` for reactive vars
- Derived: `let y = $derived(expr)` — never `$: y = expr`
- Side effects: `$effect(() => { ... })` — never `$: { ... }` blocks
- Props: `let { prop } = $props()` — never `export let prop`
- Event handlers: `onclick={fn}` — never `on:click={fn}`
- Snippets: `{#snippet name()}{/snippet}` and `{@render name()}` — never named slots

**File structure:**
- Components live in `src/lib/components/`
- Import components via the `$lib` alias: `import Foo from "$lib/components/Foo.svelte"`
- Server-side logic (data loading, backend calls) goes in `+page.server.ts`
- Client-side-only logic goes in `+page.ts`
- The browser must never call the Express backend or model service directly

**TypeScript:**
- All `.svelte` files use `<script lang="ts">`
- Use `$env/static/private` for server-only env vars (e.g. `BACKEND_URL`)
- Run `npx svelte-kit sync` after adding new env vars or routes to regenerate types

**Reference:** see `src/.example/Svelte5Reference.svelte` for a local Svelte 5 cheat sheet (not built, not a route).

---

### Backend (Express / TypeScript)

**Project structure — one file per responsibility:**
- `src/index.ts` — server startup only; no business logic
- `src/app.ts` — Express app creation, middleware registration, route mounting
- `src/config/env.ts` — all environment variables defined in one place
- `src/routes/<domain>.router.ts` — one router file per domain (e.g. `prediction.router.ts`)
- `src/routes/index.ts` — aggregates all routers; the only file that knows all route prefixes
- `src/middleware/errorHandler.ts` — all `next(err)` calls land here

**Routing conventions:**
- All API routes are prefixed `/api/v1/`
- Each router file defines routes relative to its mount point (e.g. `router.post("/")` not `router.post("/predict")`)
- Every async route handler wraps its body in `try/catch` and calls `next(err)` on failure

**TypeScript:**
- `strict: true` is enforced
- Run `npm run typecheck` (`tsc --noEmit`) before committing — no ESLint
- Use `npm run dev` in development (`tsx watch`), `npm run build` + `npm start` for production

---

### Model Service (FastAPI / Python)

**Project structure:**
- `main.py` — entry point; runs uvicorn; no business logic
- `app.py` — FastAPI app creation and router registration
- `config.py` — all environment variables; loads `.env` via `python-dotenv`
- `routers/<domain>.py` — one router file per domain (e.g. `prediction.py`)

**Routing conventions:**
- All routes are prefixed `/api/v1/` (mounted in `app.py`)
- Use Pydantic `BaseModel` for all request and response bodies — no raw `dict`
- Route functions are `async def`

**Python environment:**
- Always work inside the virtual environment at `model_service/.venv`
- Add new packages to `requirements.txt`; run `make web-install` to install
- Never install packages globally or into conda base for this project
