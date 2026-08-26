# docs/

Research and engineering documentation for the LBSM project.

- **[`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md)** — Nix flake / conda / pip setup paths, and the
  `flake.nix` shellHook heredoc bug (unquoted `<<EOF` letting backtick command substitution launch
  an interactive SageMath REPL mid-shellHook) that was diagnosed and fixed.
- **[`NB08_REAL_WORLD_GENERALIZATION_PLAN.md`](NB08_REAL_WORLD_GENERALIZATION_PLAN.md)** — task
  list for testing whether the pipeline's methodology generalises beyond the synthetic agent to
  real sequential-system telemetry without ground truth (Mars Pathfinder/Sojourner engineering
  telemetry, MSL-REMS environmental telemetry — see `datasets/README.md`).
- **[`references/`](references/Readme.md)** — codebase reference set: architecture diagrams,
  module-by-module function catalog, and implementation/design-pattern notes, verified against the
  actual code (function-level, via `ast`). Start at `references/Readme.md`'s index.

For the project's research hypothesis, current implementation status, and command reference, see
the repository root `README.md`.
