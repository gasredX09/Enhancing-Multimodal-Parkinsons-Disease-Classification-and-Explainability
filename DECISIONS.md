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
