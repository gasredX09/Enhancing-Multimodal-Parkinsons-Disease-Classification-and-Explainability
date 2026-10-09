# Multimodal Parkinson's Disease Classification and Explainability

Two versions of one project live in this repo.

| | What it is | Where |
|---|---|---|
| **v1** | The finished CMU PBAI capstone: gait, handwriting and speech models, late fusion, and SHAP explainability. Frozen as it was. | [`v1/`](v1/README.md) |
| **v2** | A redo on datasets where the same people have more than one modality, so fusion can be tested on real people and not on randomly paired ones. In planning. | [`v2/`](v2/README.md) |

## Why v2

v1 used separate public datasets for gait, handwriting and speech, with different people in each. Its fusion scores therefore come from randomly pairing people across cohorts. That is a simulation, not a multimodal test. v2 starts from datasets with the same participants in every modality, and tests fusion on them.

## Shared files

- [`AGENTS.md`](AGENTS.md): the working rules for people and AI agents in this repo.
- `CLAUDE.md`: contains only `@AGENTS.md`.
- [`DECISIONS.md`](DECISIONS.md): a dated record of structural decisions.
- [`FLOW.md`](FLOW.md): entry points and execution order for each version.
- `docs/superpowers/`: design specs and implementation plans.

## About v1's paths

v1 is kept exactly as it was, including the paths and cluster settings inside its scripts, which belong to the original course environment. Run v1 commands from inside `v1/`. The test command in `AGENTS.md` runs from the repo root. The last commit before this layout is `4c1dee4`: use `git show 4c1dee4:README.md` to see the original README, or check that commit out to see the old layout.
