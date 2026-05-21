# Contributing

## Project workflow

This project uses a research-first workflow, not a production-first workflow.

We are treating this as a learning project where progress comes from investigation, experimentation, and answering the right questions. Because of that, we are not using a strict “by the book” production task model where every branch or task must begin as a fully defined implementation unit.

For this project, every task and every branch should represent a **question**.

That question must be something that helps move the project forward.

Examples:
- Can we get this API flow working in a minimal way?
- What is the simplest structure for this feature?
- Which approach is most suitable for this part of the app?
- Can this prototype support the use case we need?

This is important because progress in this project does not come from pretending we already know the solution. Progress comes from investigating, answering questions, and turning those answers into working artifacts.

## Core principle

Every task should represent a question.

Every branch should represent a question.

Every question should be chosen because it helps move the project forward.

We are doing this deliberately to keep momentum. A rigid production workflow would slow the team down, create artificial structure too early, and make it harder to explore what actually works. Doing it “by the book” is not the right fit for this project.

## Branching approach

- `main` is the shared branch for the main application
- each team member has their own folder for exploratory work `sandbox/student_[name]`
- short-lived branches are used for investigations
- branches do not need to map to formal tickets in the traditional sense
- branches should instead reflect a line of inquiry

Branch names should reflect the question being explored or the product change being made. The prefixes we've settled on in practice:

| Prefix | Use it when… | Examples |
|---|---|---|
| `invest/<q>` | You're investigating a question — no commitment to landing the result | `invest/api-auth-flow`, `invest/can-we-use-library-x` |
| `integration/<q>` | You've answered a question and you're landing the result into `application/` | `integration/data-shape-for-search` |
| `feat/<thing>` | A product feature on the canonical web app | `feat/multi-page-flow`, `feat/model-integration`, `feat/check-in-panel` |
| `tj-<name>` / `<student>-<name>` | Personal scratch in your sandbox | `tj-roaming`, `tj-repo-file-cleanup` |

The point of the branch is to investigate something that matters or build a real feature, not to imitate a production workflow. Because we expect branches to be short-lived, create pull requests to merge your branch into `main` often — even if the work is incomplete.

## Exploratory work

Each team member may work in their own folder in the repository for uncertain, experimental, or early-stage work.

That space is for:
- prototypes
- proof-of-concepts
- tests
- rough implementations
- partial experiments
- working examples

This work does not need to be fully integrated into the main application immediately.

The purpose of exploratory work is to answer questions and produce artifacts that help the team understand what should happen next.

## From question to integration

Work should move through the project like this:

1. A question is identified.
2. A branch is created to investigate that question.
3. The investigation produces an artifact, prototype, or working example.
4. The result helps clarify what should be integrated.
5. A separate integration task or branch can then move the useful result into the main application.

This means that working artifacts are not wasted effort. They are part of how the project advances.

## What counts as a good task

A good task is not just “something to do.”

A good task is a question that:
- is clear enough to investigate
- has some chance of producing a useful answer
- helps the team make forward progress
- can produce a working artifact, insight, or decision

Bad task framing:
- build everything for feature X
- finish the whole backend
- do the UI

Better task framing:
- what backend structure supports this feature best?
- can we get a minimal version of this flow working?
- what data shape does the UI actually need?
- which implementation approach is simplest and most reliable?

## Pull requests and merges

Because this is a non-standard branching flow, it's rational to expect merge conflicts as we merge both investigation and feature branches into `main` frequently. To mitigate this:

- All investigation work should go into your `sandbox/student_<name>/` folder.
- The following are treated as **protected folders** — changes need a PR with at least one approval: `application/`, `training_pipeline/`, `report/`, `face_cropper/`, and the top-level docs (`README.md`, `ARCHITECTURE.md`, `CLAUDE.md`, `CONTRIBUTIONS.md`, `WORKTREES.md`).
- Use a `feat/` or `integration/` branch when touching protected folders. `invest/` branches should stay in sandbox until you've answered the question.

Exploratory work should only be merged into `main` when it has produced something useful enough to support the main application.

A merge should happen when:
- the investigation answered a meaningful question
- the artifact is useful to keep
- the result helps others build forward
- the team agrees it is worth integrating

## Why this workflow

We are using this workflow to keep the project moving.

This project will not move forward by forcing everything into formal production-style tasks too early. That would create overhead without helping us learn or discover what actually works.

This workflow is meant to:
- support exploration
- create momentum
- make progress visible
- turn uncertainty into usable artifacts
- help the team move from questions to working implementation

That is the standard for contribution in this project.