# Multi-Agent Development Process

This repository now follows a multi-agent workflow for implementation tasks.

## Flow

1. `agent-architect` prepares feature architecture and constraints.
2. `lead-dev-planner` decomposes work into implementation tasks.
3. `implementation-developer` applies code changes.
4. `code-review-qa` performs post-change validation and test review.
5. `docs-writer` updates docs when behavior/interfaces changed.
6. `agent-lawyer` is invoked if dependencies/licenses changed.
7. `debug-detective` is invoked for incidents and regressions.
8. `devops-engineer` owns CI/CD, packaging and release automation.

## Practical rule

- Every meaningful feature change should include:
  - implementation step,
  - QA review step,
  - documentation step.

## VoxFusion mapping

- Architecture changes: `docs/ARCHITECTURE.md`
- Task decomposition: `docs/TASKS.md`
- Binary packaging: `docs/BINARY_BUILD.md`
