# Dataset survey for v2

Date: 2026-10-08. Purpose: find datasets where the same participants have more than one modality and a Parkinson's disease (PD) label, so fusion can be tested on real people.

## How this was done

Three research agents searched public web pages and papers in parallel. They did not download data, create accounts or accept any data terms. The main session then read the key pages itself. In this document, **verified** means the main session read the claim at the source, with the URL given. **Agent-reported** means a research agent reported it from the source it cites, and it has not been checked again. Nothing here comes from memory.

## Core findings

- No open dataset was found where the same people have gait or IMU data plus speech or handwriting, with healthy controls. (agent-reported)
- None was found with speech plus handwriting from the same patients. A 2025 Scientific Reports paper (PMC12371065) says so directly. (agent-reported; the paper was not opened)
- Handwriting is therefore unlikely to be a v2 modality. The open options are voice, finger tapping and walking from one phone app (mPower), and wrist motion on 11 tasks plus a questionnaire (PADS).

## The two chosen datasets

### mPower: the main paired fusion test

- **Source:** Synapse project `syn4993293` (https://www.synapse.org/Synapse:syn4993293). Paper: Bot et al., Scientific Data 3:160011 (2016), doi 10.1038/sdata.2016.11.
- **Activities in the study** (verified on the Synapse wiki): voice, tapping, walking and a memory game, plus demographic, MDS-UPDRS and PDQ-8 surveys. Durations and sensor details are agent-reported.
- **People with all three of voice, tapping and walking** (verified, Deng et al., Communications Biology 2022, https://pmc.ncbi.nlm.nih.gov/articles/PMC8763910): "We identified a total of 2729 individuals (645 PwP) in the mPower dataset who had all three types of data." The non-PD count of 2,084 is the difference of those two numbers. The paper does not state it.
- **Label:** a self-reported professional diagnosis, never verified (verified, same paper). The target is called "self-reported PD vs not" in all v2 work.
- **Published result to compare with** (verified, same paper): a combined AUC of 0.944, with splits by person and no outside cohort. The Results section describes an average over four models, and the abstract says three.
- **Known risk** (agent-reported, from a second source): the share of PD rises steeply with age in this sample (reported as 56% at ages 50-65 and 0.95% at ages 35 and under), so age alone can leak the label. Matching or stratifying by age and sex is required.
- **Coverage** (verified, Synapse wiki): data from the first six months of the study, and only people who consented to broad sharing.
- **Access** (verified on the Synapse wiki pages "1 - Accessing the mPower data" and "5 - FAQs"):
  1. A Synapse account, with the Synapse governance policies and the Awareness and Ethics Pledge.
  2. A certification quiz of 15 questions.
  3. A validated profile: a complete profile, a public ORCID, and the signed Synapse Pledge.
  4. Identity attestation, by any one of these: a letter on letterhead from a signing official (not the applicant, and dated within the past calendar month), a notarized letter, or a professional license. Work or student ID badges and diplomas are not accepted.
  5. An intended-data-use statement, which is posted publicly, and agreement to the Conditions for Use.
  No turnaround time is stated.
- **Conditions for Use** (verified): no re-identification; keep the data confidential and secure; use it only as described in the intended-use statement; report any misuse within 5 business days; publish findings in open-access venues; acknowledge the mPower participants and study in every publication. **The data may not be redistributed.** The pages do not mention AI tools. v2 treats the data as code-only anyway (see "Controlled data and AI tools").
- The MDS-UPDRS and PDQ-8 surveys need separate approval from their copyright holders (verified). v2 does not need them.

### PADS: a parallel movement-only track

- **Source:** PhysioNet (https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/). Paper: Varghese et al., npj Parkinson's Disease 10:9 (2024), doi 10.1038/s41531-023-00625-7.
- **Verified on the PhysioNet page:** license CC BY-NC-SA 4.0; "Anyone can access the files, as long as they conform to the terms of the specified license"; 469 individuals and 5,159 measurement steps; smartwatch acceleration and rotation at 100 Hz; 11 movement tasks of 10 to 20 seconds; a questionnaire (age, height, weight, gender, kinship with PD, effect of alcohol on tremor) plus 30 yes/no non-motor symptom questions; 1.4 GB uncompressed; no speech or audio.
- **Agent-reported (from the paper):** 276 PD, 79 healthy controls and 114 people with other diagnoses, diagnosed by neurologists; balanced accuracy of 91.2% for PD against controls with nested 5-fold cross-validation and one sample per person (78.99% from movement alone, 89.79% from the questionnaire alone); the control group is mostly female and the PD group mostly male.
- **What it means:** the second "modality" is a questionnaire, not a second sensor, and the questionnaire alone nearly matches the combination. PADS suits a movement-only track, not a speech fusion test.

## Other candidates (all agent-reported, not checked)

| Dataset | What the same people have | Note |
|---|---|---|
| WearGait-PD (used in v1) | Xsens IMUs, insoles and a pressure walkway for 185 people (100 PD, 85 control) | All gait, with no speech or handwriting. CC BY 4.0, Synapse registration. An open outside gait cohort exists (MyGait, Zenodo 15672744), with different sensors. |
| PPMI | Wearable data on 353 people (148 PD, 35 control, 158 prodromal), plus imaging | Application review. Only 35 controls have watch data. Terms below. |
| WATCH-PD | Gait, tremor, tapping and speech on 82 early PD and 50 controls | Access through a consortium or a steering-committee proposal. Not planned around. |
| Rochester webcam study | Finger tapping, a smile and a spoken sentence, 845 people | The authors promise extracted features only. No download link was found. |
| NTUH smartphone study | Voice, tapping and walking, 496 people (213 PD, 283 control) | The repository linked from the paper holds code and models, and no participant data. |
| i-PROGNOSIS phone data | Phone motion plus typing | A small labeled set of 22 people, open on Zenodo. How many people appear in both data types is unverified. |

## Outside validation

Open second cohorts were found for voice only (MDVR-KCL and NeuroVoz; agent-reported). None was found for tapping or walking. So v2 results for those two modalities come from one cohort with splits by person, and say so.

## Controlled data and AI tools

PPMI's Data Use Agreement (version 5.0, April 2026; verified at https://www.ppmi-info.org/sites/default/files/docs/ppmi-data-use-agreement.pdf) says PPMI prohibits sharing participant-level data even if it is processed or altered, and that such data may only be distributed through a PPMI-approved repository. It also bars AI tools that do not guarantee containment of the data. v2 therefore treats any controlled-access data as code-only for agents: they may see code, schemas, counts and aggregate outputs, and never participant rows. The same is the default for mPower until its owners say otherwise.

## What was not checked

The mPower access turnaround and total download size; mPower per-task counts beyond the published 2,729; the WATCH-PD, Rochester and NTUH access terms; and every other agent-reported number above.
