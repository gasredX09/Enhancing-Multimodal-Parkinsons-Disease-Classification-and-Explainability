# Multimodal Parkinsons Project Plan

Goal
- Replicate the baseline multimodal pipeline using the same datasets.
- Deliver measurable improvements in alignment, evaluation rigor, fusion, and explainability.

Team Roles (4 people)
- Lead/Coordinator: integration, orchestration, final validation
- Unimodal Specialist A: speech + handwriting
- Unimodal Specialist B: gait + pseudo-labeling
- Fusion/XAI Engineer: bimodal/trimodal fusion, explainability, reporting

Phase 1: Foundation and Setup (Weeks 1-2)
Deliverables
- subject_mapping.csv (explicit subject alignment across modalities)
- data_validation_report.md (counts, missing samples, class balance)
- baseline_unimodal_metrics.md (accuracy, F1, AUC per modality)

Tasks
- Organize repo, configs, logging, and experiment tracking
- Audit datasets and build subject ID mapping
- Train speech model (EfficientNet-B0) and extract features
- Train handwriting model (ResNet-50) and extract features
- Load gait autoencoder, extract embeddings, and validate clustering

Phase 2: Alignment and Cross-Validation (Weeks 3-4)
Deliverables
- aligned_dataset.npz (subject-level aligned features)
- cv_results.json (5-fold subject-level CV metrics)
- alignment_audit_report.md (verification of ID linkage)

Tasks
- Implement subject-level alignment and validation checks
- Build stratified subject-level splits
- Implement 5-fold CV for all unimodal and fusion models
- Report mean plus or minus std for accuracy, F1, AUC

Phase 3: Fusion Improvements (Weeks 5-6)
Deliverables
- fusion_comparison_report.md
- ablation_study_results.json

Tasks
- Baseline fusion: early fusion (concatenate) and late fusion (probability weighting)
- Improvements: stacking ensemble and learned modality weights
- Feature engineering per modality (speech, gait, handwriting)
- Ablation studies: remove each modality, measure performance drop

Phase 4: Explainability and Uncertainty (Weeks 7-8)
Deliverables
- explainability_report.md
- uncertainty_analysis.json
- patient_report_templates

Tasks
- SHAP for fusion model and gait features
- Grad-CAM for handwriting, Grad-CAM++ for speech
- Add uncertainty calibration (Platt or isotonic)
- Contrastive explanations for individual predictions

Phase 5: Robustness and Generalization (Week 9)
Deliverables
- robustness_report.md
- stress_test_results

Tasks
- Missing modality tests (speech-only, gait-only, handwriting-only)
- Noise and compression tests for audio and images
- Domain shift evaluation (different recording conditions)

Phase 6: Integration and Delivery (Week 10)
Deliverables
- Final codebase, reproducibility checklist, and documentation
- Updated Streamlit demo with predictions and explanations

Tasks
- Integrate all improvements into unified pipeline
- Cleanup, docstrings, config files, reproducibility steps
- Final results tables and figures

Success Criteria
- Baseline results reproduced with clear documentation
- Subject alignment verified and audited
- Improved fusion or evaluation yields measurable gains
- Explainability outputs are interpretable and consistent
- Robustness tested under missing or degraded modalities

Immediate Next Steps
1) Build subject mapping and data audit
2) Reproduce unimodal baselines
3) Establish subject-level CV splits
4) Begin fusion baseline
