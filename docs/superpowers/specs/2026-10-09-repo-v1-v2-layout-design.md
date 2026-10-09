# Repo layout for v1 and v2: design

Date: 2026-10-09
Status: draft, waiting for review
Sub-project 0: the repo restructure. Four follow-on sub-projects are listed at the end.

## Purpose

Make room for a second version of the project ("v2") in this repo without changing or losing anything from the first ("v1", the finished capstone). After this step the repo root is short and shared, v1 sits untouched in its own folder, and v2 has a place to start.

## What was decided, and when

- 2026-10-08: v2 is a redo with better datasets and better models. "Better" means better organized and better results. The modalities are not fixed: v2 uses whichever two or three modalities have a real paired dataset, meaning the same participants in every modality. v1 scored its fusion by randomly pairing people from separate cohorts, and v2 exists to fix that.
- 2026-10-08: approach A. mPower (voice, finger tapping and walking from the same phone users) is the main paired fusion test. PADS (smartwatch motion on 11 tasks plus a symptom questionnaire) is a parallel, fully open movement track.
- 2026-10-08: v2 lives in this repo. Nothing may be destroyed.
- 2026-10-09: layout 2. v1 moves into `v1/` with `git mv`, v2 gets `v2/`, and the root holds only shared files.
- Already in force (`DECISIONS.md`, 2026-10-08): this is a public repo with no raw data in it. Carrying that out is staged and is not part of this step (see "Deferred").

## Goals

1. v1 keeps every file and every file's history. Its content does not change.
2. The root holds only what applies to both versions.
3. v2 has a home and a first document: the dataset survey that led to approach A.
4. A reader can tell in under a minute what v1 and v2 are and where each lives.
5. Every claim of "nothing lost" is checked by a command, not asserted.

## Non-goals

- No change to any v1 code, path, number or result. v1 is frozen as it is, including its stale paths and PSC account strings.
- No removal of tracked data, and no change to the PPMI files except being renamed along with their folder.
- No history rewrite, force-push, branch deletion, tag, or visibility change.
- No v2 code, environment, data download or compute.

## Target layout

```
README.md          short router: what v1 and v2 are, with links
AGENTS.md          shared rules (CLAUDE.md contains only @AGENTS.md)
CLAUDE.md
DECISIONS.md       shared, append-only
FLOW.md            shared: entry points and execution order, one section per version
.gitignore         generic rules only
docs/superpowers/  specs and plans for this work
v1/                the capstone, frozen
  README.md  CONTRIBUTING.md  requirements.txt  .gitignore
  catboost_info/  data/  docs/  notebooks/  outputs/  scripts/  src/  tests/
v2/                the redo
  README.md
  docs/dataset-survey.md
```

What moves into `v1/`: the folders `catboost_info`, `data`, `notebooks`, `outputs`, `scripts`, `src`, `tests`; every child of `docs/` except `docs/superpowers/` (today: `guidelines`, `literature`, `planning`, `roadmap`, `setup`); and the files `README.md`, `CONTRIBUTING.md`, `requirements.txt` and `.gitignore`. What stays at the root: `AGENTS.md`, `CLAUDE.md`, `DECISIONS.md` and `docs/superpowers/`.

## Procedure

The work is three commits, made back to back and pushed together, so `main` is never visible with ignore rules that no longer match.

**Pre-flight (read-only).**
1. The working tree is clean, the branch is `main`, and it equals `origin/main`.
2. Record the HEAD hash, the git hash of every path that will move, the count of tracked files, and the list of ignored local files (`git status --ignored --short`).
3. Run the tests. Six pass today.

**Commit 1, "Move the v1 capstone into v1/".** Only `git mv`, with no content edits mixed in, so git records pure renames. Moving a folder with `git mv` carries ignored local files along with it.

**Commit 2, "Add the root README, split .gitignore, update shared files".**
- A new root `README.md` that explains the two versions and links to `v1/README.md` and `v2/README.md`. The old README moved unchanged in commit 1.
- `.gitignore` is split. The root keeps the generic rules: `.DS_Store`, Python cache and build folders, local environments (`.env`, `*.local.*`, virtual environments), editor and Jupyter folders, logs, model weight and array files (`*.pt`, `*.pth`, `*.pkl`, `*.joblib`, `*.npy`, `*.npz`), and archives. It also gains `v2/data/` as a safety net, because v2 data lives outside the repo. `v1/.gitignore` keeps only the rules that name v1 paths: the embeddings exception for `.npz` files, the speech `catboost_info` folder, the generated handwriting artifacts, and the `data/*` allowlist. Rules in a deeper `.gitignore` override the root, so the embeddings exception still works. A check with `git check-ignore` confirms each rule still behaves as before.
- `AGENTS.md`: the test command becomes `python3 -m unittest discover -s v1/tests -v`.
- `FLOW.md` is created. Its v1 section lists each pipeline's entry points and execution order (figshare IMU gait, WearGait gait, handwriting, speech, late fusion), read from the code at the time of writing, with anything not verified marked as unverified. The v2 section is added when v2 code exists.
- `DECISIONS.md` gets two new entries, dated by the day each decision was made: v2 in this repo with approach A (2026-10-08), and the v1/v2 layout with nothing destroyed and data removal staged, archive first (2026-10-09). Existing entries are not edited.

**Commit 3, "Start v2".** `v2/README.md` (status: planning; approach A; the data sources and where to read their terms) and `v2/docs/dataset-survey.md`. The survey separates what was read at the source from what a research agent reported, lists each candidate with its access path, and states that the target label for mPower is self-reported PD. It contains no participant-level data.

**Push.** One normal `git push origin main` of the three commits. No force.

## Verification

After commit 1, and again after commit 3, all of these must hold. Any failure stops the work.

1. For every moved path, the git hash at the old path in the pre-flight record equals the hash at its new `v1/` path. Identical hashes mean identical content, names and file modes.
2. `git diff --name-status -M HEAD~1 HEAD` shows only `R100` entries (100% renames) for commit 1: no additions, deletions or modifications.
3. The count of tracked files is the same before and after commit 1.
4. The six existing tests pass from the new location with the new command.
5. Every ignored local file recorded in pre-flight exists under `v1/`, and none remain at its old path.
6. `git log --follow` on a sample of moved files (one per top-level folder) reaches the original commits.
7. `git status` is clean, and `git check-ignore` gives the expected result for a sample of paths: an ignored path under `v1/data/`, the embeddings exception, a `*.local.*` file at the root, and a path under `v2/data/`.
8. The links in the root README resolve.
9. The new files pass the style scan (no em dashes, no double hyphens as punctuation) and the privacy scan (no personal paths, account IDs, emails or token shapes).

## Rollback

Undo with `git revert` of the commits, which creates new commits. No reset, no amend, no force-push. Nothing is deleted from history, so the original layout can always be restored by reverting.

## Effects on other people and systems

- Teammates who pull will see the folders renamed. Ignored local files inside the old folders (data they downloaded themselves) stay where they are, because git only moves tracked files on a pull. Nothing is deleted. They can move those files into `v1/` by hand.
- The other remote branches are not touched.
- Links that point to old file paths on GitHub, in the course report or elsewhere, stop resolving. The `v1/` paths replace them.
- Any copy of the repo already deployed on PSC is unaffected. This step does not touch PSC.

## Deferred (each needs its own approval)

1. **Untracking v1's data (about 1.7 GB).** Untracking with `git rm --cached` deletes the files from the working tree of anyone who pulls. First the data is copied to an archive outside the repo with a SHA-256 manifest, and only then can untracking be proposed.
2. **The PPMI files.** Handling is decided separately, after the owner answers who obtained them and under what agreement.
3. **History.** Any rewrite, force-push, branch deletion, `.mailmap`, tag or visibility change.
4. **Scrubbing v1's hardcoded PSC account strings and machine paths** from its current files. This conflicts with freezing v1, so it is its own decision.
5. **Credential rotation.** The repo owner does this.

## Risks

- A mistake in the move, such as a missed folder. The hash comparison in "Verification" catches it before the push.
- Hidden references to old paths in v1 files. They stay stale on purpose, and the root README says so.
- `FLOW.md` could describe a pipeline wrongly. It is written from the code, and unverified items are marked.
- The root `.gitignore` could miss a rule the old one had. The `git check-ignore` samples and a before-and-after comparison of the ignore rules cover this.

## Acceptance criteria

The verification list passes, the root contains only the shared files plus `v1/`, `v2/` and `docs/superpowers/`, the tests pass, and `main` holds the three commits as a normal fast-forward.

## Follow-on sub-projects

Each gets its own spec and plan, in this order:
1. Data access and manifests. The mPower application is the owner's to submit. PADS is downloaded and checksummed.
2. Shared infrastructure: a locked environment, a run record for every job, and PSC deploy, submit and fetch with account values in a local ignored file.
3. Per-modality models on mPower (voice, tapping, walking) and PADS (movement).
4. Fusion and explainability on people who really have every fused modality, with the evaluation protocol in `AGENTS.md`.
