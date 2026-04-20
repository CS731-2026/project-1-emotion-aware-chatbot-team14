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

Branch names should reflect the question being explored.

Examples:
- `invest/api-auth-flow`
- `invest/dashboard-layout-options`
- `invest/data-shape-for-search`
- `invest/can-we-use-library-x`

The point of the branch is to investigate something that matters, not to imitate a production workflow. 
Because we expect branches to short lived, create pull requests to merge your branch into `main` often. Even if the work is incomplete.

Or branches should reflect the integration of results of inquiry into the main application logic.

- `integration/api-auth-flow`
- `integration/dashboard-layout-options`
- `integration/data-shape-for-search`
- `integration/can-we-use-library-x`

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
Because this is a non standard branching flow, its rational to expect merge conflicts as we merging both investigation and integration branches into the main branch frequently. To mitigate this, all our investigation work should go into our `sandbox/student_[name]` folders, and the following folders: `application`, `training_pipeline` and `report` are treated as the main application folders. Any changes to the main application folders needs to be a `integration` branch and PRs need to approved to control the quality of the main code.

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