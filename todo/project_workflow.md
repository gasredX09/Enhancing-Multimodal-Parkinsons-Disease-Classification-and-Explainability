# PROJECT WORKFLOW: MULTIMODAL PARKINSON'S DISEASE CLASSIFICATION WITH IMPROVEMENTS

I. PROJECT VISION

Goal: Replicate the baseline multimodal pipeline while implementing 3 to 4 key improvements that enhance accuracy, robustness, and clinical interpretability.

Success Metrics:

- Reproduce baseline performance (85 percent or higher accuracy, greater than 0.90 AUC)
- Improve at least 2 key areas (cross-validation, data alignment, feature engineering, or ensemble methods)
- Deliver production-ready code with full documentation
- Generate actionable insights via explainability analysis

II. TEAM STRUCTURE (4 PEOPLE)

Role | Person | Responsibilities
Lead (Coordinator) | Person 1 | Pipeline orchestration, integration, final validation
Unimodal Specialist 1 | Person 2 | Speech + Handwriting models
Unimodal Specialist 2 | Person 3 | Gait model + pseudo-labeling
Fusion + XAI Engineer | Person 4 | Bimodal + Trimodal fusion, explainability

III. PROJECT PHASES AND TIMELINE

PHASE 1: FOUNDATION AND SETUP (WEEKS 1-2)

Deliverable: Reproducible baseline, organized codebase, data validation report

Week 1: Data Preparation and Code Organization

- Person 1:
 	- Create modular codebase structure (config management, logging, error handling)
 	- Build data loader framework (with explicit subject ID mapping)
 	- Setup version control + documentation templates

- Person 2 and 3:
 	- Download all three datasets
 	- Create subject mapping CSV (Speech ID <-> Gait ID <-> Handwriting file)
 	- Data quality audit: check for missing samples, labeling errors
 	- Generate dataset statistics (sample counts, class balance, modality dimensions)

- Person 4:
 	- Setup experiment tracking (MLflow or Weights and Biases)
 	- Create visualization templates for results

Deliverable: data_validation_report.md, subject_mapping.csv, modular config.py

Week 2: Reproduce Baseline Unimodal Models

- Person 2: Train EfficientNet-B0 for speech
 	- Implement audiomentations (data augmentation)
 	- Track training curves, save best model
 	- Extract 1280D feature vectors for all speech samples

- Person 3: Train ResNet-50 for handwriting
 	- Implement image augmentations (rotation, affine)
 	- Extract 2048D feature vectors
 	- Compare pre-trained vs random initialization

- Person 3 (continued): Load pre-trained TCN autoencoder for gait
 	- Extract embeddings for all gait windows
 	- Validate clustering quality (silhouette scores)
 	- Extract 18D features from bottleneck

- Person 4: Create baseline feature extraction pipeline
 	- Standardize feature dimensions
 	- Create feature metadata (timestamps, subject IDs)

Deliverable: results/baseline_unimodal_metrics.md (accuracy, precision, recall, AUC for each modality)

PHASE 2: IMPROVEMENT 1 - DATA ALIGNMENT AND CROSS-VALIDATION (WEEKS 3-4)

Improvement Focus: Fix subject misalignment issue + implement k-fold cross-validation

Deliverable: Aligned dataset, CV results, data robustness analysis

Week 3: Data Alignment and Stratified Splitting

- Person 1:
 	- Implement subject-level data alignment (trace Speech <-> Gait <-> Handwriting via IDs)
 	- Build alignment validation function (verify same subject gets same label across modalities)
 	- Create stratified subject-level train/test split (not sample-level)
 	- Generate alignment audit report with verification visualizations

- Person 2 and 3:
 	- Validate that their feature extraction respects subject boundaries
 	- Create subject-aggregated features (per-subject mean + std)

Deliverable: aligned_dataset.npz, alignment_audit_report.md

Week 4: Cross-Validation Framework

- Person 1:
 	- Implement 5-fold stratified subject-level k-fold CV
 	- Create CV split generator (maintains subject integrity)
 	- Setup automated model training across all folds

- Person 4:
 	- Track metrics per fold: mean plus or minus std for accuracy, AUC, F1, precision, recall
 	- Generate cross-validation stability plots
 	- Statistical significance testing (paired t-tests across folds)

Deliverable: cv_results.json, cv_stability_plots, improved baseline metrics with confidence intervals

PHASE 3: IMPROVEMENT 2 - ENHANCED FUSION STRATEGIES (WEEKS 5-6)

Improvement Focus: Move beyond basic early fusion to attention-based and ensemble fusion

Deliverable: Novel fusion models, ablation study, fusion comparison report

Week 5: Multimodal Fusion Enhancements

- Person 4:
 	- Stacking ensemble: Train meta-learner on stacked predictions from bimodal models
 	- Feature weighting: Learn importance of each modality via attention mechanism
 	- Late fusion with calibration: Optimize fusion weights for each fold using Platt scaling

Code structure:

```python
# Option A: Learned fusion weights
X_fused = w_speech * speech_features + w_gait * gait_features + w_hw * hw_features
# Optimize w_1, w_2, w_3 on validation set

# Option B: Stacking
meta_features = stack([clf_speech(X_speech), clf_gait(X_gait), clf_hw(X_hw)])
meta_clf = LogisticRegression().fit(meta_features, y)

# Option C: Attention mechanism
attention_weights = softmax(dense_layer([speech, gait, hw]))
fused = sum(attention_weights[i] * modality[i] for i in [speech, gait, hw])
```

- Person 2 and 3:
 	- Feature engineering: Extract domain-specific markers
  		- Speech: F0 contour, jitter, shimmer statistics
  		- Gait: Stride length variability, cadence regularity, bilateral asymmetry
  		- Handwriting: Pressure distribution, stroke velocity profiles
 	- PCA or ICA analysis: Identify independent components per modality

Week 6: Ablation and Comparison Study

- Person 4:
 	- Compare all fusion strategies:
  		- Baseline (early fusion)
  		- Late fusion (prob averaging)
  		- Stacking
  		- Learned weights
  		- Attention mechanism
 	- Ablation study: Remove each modality, measure accuracy drop
 	- Bimodal contribution analysis (Speech + Gait vs Speech + Handwriting vs Gait + Handwriting)

- Person 1:
 	- Create comparison table + visualizations
 	- Statistical significance testing (ANOVA on CV folds)

Deliverable: fusion_comparison_report.md, ablation_study_results.json, trained ensemble models

PHASE 4: IMPROVEMENT 3 - ENHANCED EXPLAINABILITY AND VALIDATION (WEEKS 7-8)

Improvement Focus: Beyond SHAP to contrastive explanations, uncertainty quantification, clinical validation

Deliverable: Explainability report, uncertainty estimates, clinical interpretation guide

Week 7: Advanced XAI and Uncertainty

- Person 4:
 	- LIME (Local Interpretable Model-agnostic Explanations):
  		- Explain individual predictions with local approximations
  		- Identify which features drive specific patient classification

 	- Contrastive explanations:
  		- Why PD and not HC? What would change to flip prediction?
  		- Perturbation-based explanations

 	- Uncertainty quantification:
  		- Monte Carlo Dropout: Run model multiple times with dropout enabled
  		- Confidence calibration: Platt scaling or isotonic regression
  		- Generate confidence intervals for predictions

 	- Feature interaction analysis:
  		- SHAP interaction plots (which features interact)
  		- PDP (Partial Dependence Plots) for top features

- Person 2 and 3:
 	- Domain validation: Do learned important features match clinical knowledge?
  		- Speech: Does model weight vocal tremor or dysarthria markers?
  		- Gait: Does model focus on stride irregularity metrics?
  		- Handwriting: Does model detect tremor patterns?

Week 8: Clinical Validation and Reporting

- Person 4:
 	- Generate per-patient reports (prediction + confidence + key explanations)
 	- Create confusion matrix analysis (False Positives, what patterns)
 	- ROC or PR curves with operating point recommendations
 	- Sensitivity or specificity trade-off analysis for different thresholds

- Person 1:
 	- Compile final validation report
 	- Create clinical decision support guidelines

Deliverable: explainability_report.md, patient_report_templates, uncertainty_analysis.json, clinical guidelines

PHASE 5: IMPROVEMENT 4 - ROBUSTNESS AND GENERALIZATION (WEEK 9)

Improvement Focus: Test robustness to data variations, missing modalities, domain shift

Deliverable: Robustness benchmark, domain adaptation analysis

Week 9: Robustness Testing

- Person 1 and 4:
 	- Missing modality scenarios: What if one modality is unavailable?
  		- Train fallback models: Speech-only, Gait-only, Handwriting-only
  		- Create missing-modality imputation strategies

 	- Data corruption: Add noise, compress audio, degrade image quality
  		- Test accuracy degradation curves

 	- Domain shift: What if audio quality changes? Different camera for handwriting?
  		- Adversarial perturbations
  		- Domain adversarial training (optional, if time allows)

 	- Synthetic data augmentation: Mixup, CutMix for multimodal data
  		- Train model on partially synthetic data
  		- Measure generalization impact

- Person 2 and 3:
 	- Contribute modality-specific stress tests

Deliverable: robustness_report.md, stress_test_results

PHASE 6: INTEGRATION AND DEPLOYMENT (WEEK 10)

Deliverable: Polished codebase, documentation, deployment package

Week 10: Final Integration

- Person 1:
 	- Integrate all improvements into single unified pipeline
 	- Create inference wrapper (one function: load data to predict to explain)
 	- Code cleanup: remove debug code, add docstrings, type hints
 	- Create reproducibility checklist (random seeds, versioning)

- Person 4:
 	- Update Streamlit dashboard with new features (uncertainty, advanced XAI)
 	- Add batch prediction mode

- All:
 	- Comprehensive documentation:
  		- README.md: Quick start guide
  		- METHODOLOGY.md: Detailed methods
  		- RESULTS.md: Final performance metrics
  		- IMPROVEMENTS.md: What was improved and why
  		- USER_GUIDE.md: How to use the system
  		- REPRODUCTION_GUIDE.md: Step-by-step to reproduce all results

Deliverable: Production-ready codebase, full documentation

IV. KEY IMPROVEMENTS SUMMARY

Improvement | Week | Impact | Difficulty
Data Alignment + k-Fold CV | 3-4 | Robust metrics, subject integrity | Medium
Better Fusion Strategies | 5-6 | Plus 2 to 5 percent accuracy, understanding | Medium-High
Advanced XAI | 7-8 | Clinical interpretability | Medium
Robustness Testing | 9 | Production readiness | Medium

V. DELIVERABLES CHECKLIST

Code Artifacts

- Modular Python package (parkinsons_ml)
- Configuration system (config.yaml)
- Data loader with subject alignment (data_loader.py)
- Unimodal trainers (3 scripts)
- Bimodal or Trimodal fusion (2 scripts)
- k-Fold CV orchestrator
- Advanced XAI module (LIME, SHAP, uncertainty)
- Inference pipeline
- Streamlit dashboard (enhanced)

Documentation

- README (installation, quick start)
- METHODOLOGY (detailed methods)
- RESULTS (performance metrics, tables, figures)
- IMPROVEMENTS (what got better, why)
- USER_GUIDE (how to use)
- REPRODUCTION_GUIDE (exact steps to reproduce)

Data Artifacts

- subject_mapping.csv (subject alignments)
- data_validation_report.md
- Trained models (unimodal + fusion)
- Feature vectors + embeddings (for reproducibility)
- Cross-validation splits (fold definitions)

Analysis Artifacts

- Performance comparison table
- Ablation study results
- Feature importance rankings
- Calibration plots
- ROC or PR curve comparisons
- Robustness test results
- Patient report examples

Presentation Materials

- Project proposal slide deck
- Methodology poster
- Results summary slide
- 2 to 3 minute demo video (inference + explanation)

VI. SUCCESS CRITERIA (YOUR PROJECT PROPOSAL TALKING POINTS)

Reproducibility

- All results reproducible with fixed random seeds
- Subject-level alignment fully documented
- Data split reproducible across runs

Improvement Over Baseline

- k-Fold CV reduces uncertainty
- Better fusion adds 2 to 5 percent accuracy
- Advanced XAI provides clinical insights
- Robustness testing ensures generalization

Clinical Relevance

- Subject integrity maintained (no data leakage)
- Uncertainty quantification for risk stratification
- Explainable predictions for physician trust
- Handles missing modality scenarios

Code Quality

- Modular, testable, documented
- Follows ML best practices
- Production-ready (error handling, logging)

VII. WEEK-BY-WEEK EXECUTION SUMMARY

Week 1-2: Foundation (Data, baseline unimodal models)
Week 3-4: Alignment + CV (Fix data issues, robust evaluation)
Week 5-6: Better fusion (Ensemble methods, feature engineering)
Week 7-8: Advanced XAI (LIME, uncertainty, clinical validation)
Week 9: Robustness (Stress testing, missing modalities)
Week 10: Integration (Final code, documentation, polish)

VIII. PRESENTATION FRAMING FOR YOUR PROPOSAL

Title: Multimodal Parkinson's Disease Classification: Replication with Robustness and Interpretability Improvements

Key Selling Points:

1. Clinical rigor: Fixed data alignment issue present in baseline
2. Robust evaluation: k-Fold cross-validation instead of single split
3. Better models: Learned fusion weights + ensemble methods
4. Trustworthy AI: Advanced explainability + uncertainty quantification
5. Production ready: Tested for robustness, missing modalities, domain shift
