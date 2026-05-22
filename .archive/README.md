# `.archive/`

Cold storage for code that was useful at some point but is no longer in
the active flow of the project. Files here aren't imported by anything
under `application/`, but the history and the code itself are preserved
so we can lift snippets / patterns back out when needed.

Contents follow the same shape they had at the top level — a moved
`foo/` lives as `.archive/foo/`. Use `git log --follow` on a file path
to walk its history across the move.

## What's currently archived

- **`training_pipeline/`** — the YAML-config + step-persistence training
  harness from April. Solid design (typed steps, content-addressed
  artifact store, replayable runs) but too much surface area to onboard
  the team onto in time for presentation. Replaced on
  `training-pipeline-v2` with something simpler.
