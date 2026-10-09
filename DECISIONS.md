# Decisions

Append-only and dated. An entry is never rewritten. A reversal is a new entry
that says `**Supersedes:**` and names the old one.

## 2026-10-08: Public showcase repo with no raw data

**Decision:** This repo stays public as a showcase and reproducible record of the
42-657 capstone. Raw datasets leave the repo. In their place the repo carries a
description of where each dataset comes from, how to get access, and a
checksum manifest of the expected files. Status: decided, not yet carried out.
The data is still tracked today, and old commits still contain it.
**Options considered:** Make the repo private and leave everything as is. Optimize
only for a teammate rerunning the pipeline on PSC.
**Why:** The goal is a visible, reproducible project. Several datasets have
access terms (WearGait-PD needs a Synapse account and agreement to the Synapse
pledge), so the repo should point to them and not host copies.

## 2026-10-08: Synapse token and data root come from the environment

**Decision:** `data/gait/weargait/download_weargait_pd_v1.py` reads the Synapse
token only from `SYNAPSE_AUTH_TOKEN` and writes into `$PD_DATA_ROOT/gait/weargait`.
It exits with a clear message if either variable is missing. The two
`SYNAPSE_METADATA_MANIFEST.tsv` files now hold paths relative to
`$PD_DATA_ROOT/gait/weargait`, with no absolute machine or account paths.
Regression tests are in `tests/test_download_weargait.py`, run with
`python3 -m unittest discover -s tests -v`.
**Options considered:** Fall back to cached `synapse login` credentials when the
variable is missing. Default the data root to the repo's own `data/` folder.
Use pytest for the tests.
**Why:** An earlier version of the script contained a Synapse personal access
token. It is still in git history, so it has to be revoked in Synapse. Deleting
the line does not protect it. Failing explicitly avoids hidden defaults, and the
script will move when the layout is reorganized, so a location default would go
stale. `unittest` is in the standard library, so no new dependency is added
before the environment is settled.

## 2026-10-08: v2 is a redo on paired-modality data, in this repo

**Decision:** Build a second version of the project ("v2") in this repo. v2 uses whichever two or three modalities have a real dataset in which the same participants appear in every modality. The plan is approach A: mPower (voice, finger tapping and walking from the same phone users) as the main paired fusion test, and PADS (smartwatch motion on 11 tasks plus a symptom questionnaire) as a parallel, fully open movement track. The mPower target is self-reported PD and is called that everywhere. Nothing from v1 may be destroyed.
**Options considered:** Keep the three v1 modalities (gait, handwriting, speech) from separate datasets and keep scoring fusion on randomly paired people. Use only fully open data (PADS and WearGait-PD). Put v2 in a new repo.
**Why:** v1's fusion scores come from randomly pairing people across separate cohorts, so they are a simulation and not a multimodal test. No open dataset was found with the same people in gait plus speech or handwriting, so the data decides the modalities. mPower is the largest set found with three modalities from the same people (2,729 people with all three, in the published analysis). A new repo was considered and not chosen. The dataset survey is in `v2/docs/dataset-survey.md`.

## 2026-10-09: v1 moves into v1/ and v2 gets v2/ (layout 2)

**Decision:** v1 moves unchanged into `v1/` with `git mv`, so every file keeps its history. v2 lives in `v2/`. The repo root holds only shared files: the router `README.md`, `AGENTS.md`, `CLAUDE.md`, `DECISIONS.md`, `FLOW.md`, `.gitignore` and `docs/superpowers/`. Nothing is deleted, history is not rewritten, and nothing is force-pushed. Removing v1's tracked data (about 1.7 GB) from the working tree is not part of this step. It happens only after the data is copied to an archive outside the repo with a SHA-256 manifest, and it needs its own approval. The 2026-10-08 decision "Public showcase repo with no raw data" stays in force, and this entry stages how it is carried out.
**Options considered:** Add `v2/` and leave v1 where it is. Tag the current state, let v2 take over the root, and archive v1 under `archive/v1/`.
**Why:** Moving v1 as one block keeps it complete and browsable and gives the root a short list. Git records the move as renames, and a script compares the hash of every file before and after to prove nothing changed. Untracking data deletes files from the working tree of anyone who pulls, so it waits for a safe copy. Design: `docs/superpowers/specs/2026-10-09-repo-v1-v2-layout-design.md`.
