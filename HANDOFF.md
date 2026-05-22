# Handoff — `feat/multi-page-flow`

Picking this branch up cold. Read top-to-bottom once before touching anything.

## Goal

Turn the empathy bot from a single-screen chat into a multi-page flow where
the **reasoner decides where the conversation goes next** (mode + stage),
and the frontend renders different views per mode. The user never sees the
machinery — transitions are internal.

```
mode:  qa  ──▶  feedback  ──▶  qa  ──▶  …  ──▶  done
                                   (consent gates the whole thing, Phase 7)

qa stages:  open → explore → ground → close
```

## Worktree location & ports

- Worktree: `../hri-worktrees/feat--multi-page-flow/`
- Branch: `feat/multi-page-flow`
- Ports (from local `.env.ports`, gitignored): backend `3021`, frontend `5193`, model `8020`
- `make dev` auto-picks up these ports via `-include .env.ports` in the Makefile.

The Makefile/scripts that make `.env.ports` work are still uncommitted on
`codex/empathy-bot-ui` at time of writing. They already function locally
because `make` reads the file regardless; just don't expect a fresh clone
from `main` to have port isolation yet.

## Phase roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Thread `mode` + `stage` through reasoning pipeline. Replace single `SYSTEM_PROMPT` with `BASE_PERSONA` + `MODE_PROMPTS` + `STAGE_PROMPTS`. End-to-end plumbing frontend → Express → FastAPI → `LLMReasoningAgent`. | ✅ `cb7bc085` |
| 2 | Reasoner **decides** transitions: emits `{reply, next_mode, next_stage}` JSON. Frontend `$state` store mirrors them and routes views by mode. Stubs for `feedback` / `consent` / `done`. Dashboard stub. Soft `TURN_CAP=30`. | ✅ `f3fc92e6` |
| 3 | (open) Decide whether `qaTurnCount` / `feedbackTickIndex` need to feed back into the prompt context, or stay UI-only. Currently incremented but unused by reasoner. | ⏳ |
| 4 | `/feedback` route on model_service: user reports felt emotion, server compares to predicted, writes `FeedbackEvent` to a session buffer + per-profile jsonl log. Frontend `feedbackEvents[]` mirrors it (authoritative copy stays server-side). | ⏳ |
| 5 | Timed self-report UI replacing the `feedback` mode stub in `+page.svelte`. Ticks N times then transitions back to `qa` (or `done`). | ⏳ |
| 6 | Staff dashboard at `/dashboard` (currently a stub). Password gate + aggregated feedback events + conversation metrics. | ⏳ |
| 7 | Consent flow replacing the `consent` mode stub. Profile select happens here too. Gates entry into `qa`. | ⏳ |

## What works right now

End-to-end on a real LLM provider (Gemini or OpenAI). The stub branch in
`routers/chat.py` does **not** exercise transitions — it always echoes
the caller's mode/stage back.

- Frontend sends `{text, mode, stage}` → Express forwards → FastAPI runs
  `LLMReasoningAgent.reason` → returns `ReasoningResult` → router serializes
  `{response, next_mode, next_stage, debug}` → Express forwards → frontend
  mirrors into `conversationState` store → view rerenders.
- `parse_reasoning_output` is **forgiving**: bad JSON, missing keys, or
  unknown mode/stage values fall back to the raw text with no transition.
  This is intentional — a bad reasoner turn should not crash the chat.
- `TURN_CAP = 30` in `routers/chat.py` is a hard safety: after that many
  user turns the response is rewritten to `next_mode="done"` regardless of
  what the reasoner said. Tune in code, not env, for now.

## Key files (Phase 2 surface area)

```
application/model_service/
  core/llm/reasoning_agent.py   ← OUTPUT_INSTRUCTIONS, ReasoningResult,
                                  parse_reasoning_output, reason()
  routers/chat.py               ← ChatResponse {next_mode, next_stage},
                                  TURN_CAP, stub branch preserves state

application/backend/src/routes/
  chat.router.ts                ← forwards transitions, validates Mode/Stage,
                                  defaults to caller state on model_service down

application/frontend/src/
  lib/api.ts                    ← sendChat response type
  lib/conversation/store.svelte.ts  ← $state store: mode, stage, qaTurnCount,
                                      feedbackTickIndex, feedbackEvents[]
  routes/+page.svelte           ← {#if mode === "qa"} hero, stubs for
                                  feedback / consent / done
  routes/dashboard/+page.svelte ← Phase 6 stub
```

## How to test locally

1. `cd ../hri-worktrees/feat--multi-page-flow`
2. Ensure `application/model_service/.env` has `LLM_PROVIDER` + key (Gemini
   or OpenAI). Without a real provider you fall into the stub branch and
   transitions never fire.
3. `make install` (first time only), then `make dev`.
4. Open `http://localhost:5193`.

**Checks worth doing:**

- **Happy path** — converse; watch FastAPI logs for `next_mode` / `next_stage`.
  Should walk `open → explore → ground → close` over many turns.
- **Mode stubs render** — temporarily set the store's initial `mode` to
  `"feedback"`, `"consent"`, or `"done"` in `store.svelte.ts`, refresh,
  confirm the right stub shows. Revert.
- **Bad-JSON fallback** — temporarily weaken `OUTPUT_INSTRUCTIONS` so the
  LLM emits prose. `parse_reasoning_output` should log a warning and the
  chat should keep working with no transition. Revert.
- **Turn cap** — drop `TURN_CAP` to 3, send 3 turns, confirm forced
  `next_mode="done"` and the `done` stub renders. Revert.

## Running alongside other worktrees

Two worktrees can run concurrently because each has its own `.env.ports`.
Two gotchas:

- **Session cookies are per-host, not per-port.** Logging into a profile on
  `:5193` can clobber `:5173`'s `req.session.profileId`. Use different
  browsers / incognito if you need both.
- **`data/profiles/` is per-worktree.** Profiles don't sync across worktrees.
  That's fine for testing — just don't be surprised.

## Decisions worth knowing

- **Reasoner-driven transitions, not state-machine-driven.** The LLM picks
  `next_mode` / `next_stage` every turn. The frontend store is a mirror,
  not the source of truth. Pros: simple, prompt-tunable. Cons: needs the
  `TURN_CAP` safety net and the parser fallback.
- **Frontend store is `$state`, not a class.** Svelte 5 runes only — see
  `CLAUDE.md`. Don't introduce a store class.
- **`feedbackEvents[]` on the client is a display mirror.** Authoritative
  copy will live in the model_service session + per-profile jsonl log
  (Phase 4). Don't make the client write authoritative state.
- **`/api/v1/chat` still returns `response` (not `reply`).** The wire
  format kept the old field name to avoid churning the Express route /
  frontend api type; only the internal `ReasoningResult` uses `reply`.

## Suggested next step

Phase 4 is the highest-value unblock — once feedback events are being
captured, Phase 5 (self-report UI) and Phase 6 (dashboard) both have real
data to render. Phase 3 (whether `qaTurnCount` etc. feed back into the
prompt) is a small design decision that can wait.
