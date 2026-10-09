# v2: Parkinson's classification on paired-modality data

Status: planning. There is no code yet.

## Goal

v1 trained models on separate cohorts, so its fusion scores come from randomly pairing people across them. v2 uses datasets where the same people have more than one modality, and tests fusion on those people.

## Plan

- **mPower** (Sage Bionetworks, on Synapse): voice, finger tapping and walking from the same phone users. The published analysis found 2,729 people with all three, of whom 645 reported a PD diagnosis. The labels are self-reported, so the target is called "self-reported PD vs not" everywhere. Access needs a Synapse account, a certification quiz, identity attestation (a notarized letter, a letter from a signing official, or a professional license) and an intended-use statement. The data cannot be redistributed.
- **PADS** (PhysioNet): smartwatch motion on 11 tasks plus a symptom questionnaire for 469 people, openly downloadable under CC BY-NC-SA 4.0. It has no speech, so it serves as a movement-only track.

Why these two, and what else was considered: [`docs/dataset-survey.md`](docs/dataset-survey.md).

## Ground rules

The rules in [`../AGENTS.md`](../AGENTS.md) apply. In particular: no raw or participant-level data in git, the data root comes from `PD_DATA_ROOT`, splits are by person, and fusion is tested only on people who have every fused modality.

## Next

Separate specs and plans, in this order: data access and manifests; shared infrastructure; per-modality models; fusion and explainability.
