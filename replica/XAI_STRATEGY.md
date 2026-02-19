# Explainability (XAI) Strategy for Multimodal Parkinson's Prediction

This document outlines the recommended explainability methods for each modality and fusion level in the replica project.

---

## Core Principle

Use **consistent, robust methods** across modalities to enable direct comparison and improved interpretability in a medical context.

---

## Unimodal Explainability

### Speech (EfficientNet on Mel-Spectrograms)

**Primary Method: `SmoothGradCAMpp`**
- Generates attention maps on mel-spectrograms showing which frequency bands and time steps are important
- Robust to noise and handles multiple discriminative regions well
- Implementation: Use `torchcam.methods.SmoothGradCAMpp` on the last convolutional layer
- Visualize by overlaying on the mel-spectrogram with a jet colormap

**Secondary Method: SHAP on Pooled Embeddings**
- If you export CNN feature vectors (e.g., from the global pool before classification layer), apply SHAP to show per-feature contributions
- Provides a summary-level view of which acoustic characteristics (pitch, energy, etc.) drive predictions

**Occlusion Sensitivity (Optional)**
- Mask contiguous time-frequency regions and measure probability drop
- Shows which acoustic bands are most critical for the model's decision

---

### Handwriting (ResNet-50 on Spiral/Wave Images)

**Primary Method: `SmoothGradCAMpp`**
- Shows which regions of the spiral/wave drawing the model focuses on (tremors, irregularities)
- More robust than basic Grad-CAM for capturing multiple areas of interest (strokes, curves)
- Implementation: Use `torchcam.methods.SmoothGradCAMpp` on ResNet's layer4[-1].conv3
- Overlay on the original image with alpha blending for clear visualization

**Secondary Method: Integrated Gradients**
- Provides per-pixel importance scores along a baseline (blank image or Gaussian noise)
- More theoretically grounded than gradient-based methods; good for publications
- Implementation: Use `captum.attr.IntegratedGradients` on the image input

**Occlusion Sensitivity (Optional)**
- Mask patches of the image and measure prediction change
- Highlights critical stroke regions and drawing features

---

### Gait (Autoencoder Embeddings + Classifier)

**Primary Method: SHAP on Feature Vectors**
- Apply `shap.TreeExplainer` if using a tree-based classifier (Random Forest, XGBoost)
- Or use `shap.DeepExplainer` if classifier is a neural network
- Shows which gait features (joint angles, velocities, symmetry indices, etc.) are most predictive

**Secondary Method: Temporal Saliency / Channel Importance**
- Occlude sliding windows across time steps (e.g., 10-step windows) and measure probability drop
- Identify critical gait phases (stance, swing, turning)
- Occlude individual sensor channels/axes to show which dimensions matter most

**Tertiary Method: Integrated Gradients on Embeddings**
- If using deep embeddings from the autoencoder, compute IG on the embedding layer
- Traces back to which raw sensor values contributed to the learned representation

---

## Bimodal Explainability

### Speech + Gait

**Modality Ablation**
- Compute predictions with speech zeroed → shows gait-only contribution
- Compute predictions with gait zeroed → shows speech-only contribution
- Visualize as a bar chart: original prediction confidence, speech-only, gait-only
- Intuitive for clinicians: "What if patient had speech disorder only?"

**Grouped SHAP on Fused Vector**
- Apply SHAP TreeExplainer on the concatenated (speech_features, gait_features) vector
- Color/group by modality (e.g., speech features in blue, gait in orange)
- Shows which individual features push the decision and their modality origin

**Consistency Checks**
- Do both modalities's top features agree on the class label?
- Highlight instances where modalities disagree (might be interesting edge cases or errors)

---

### Handwriting + Gait

**Side-by-Side Visual + Tabular**
- Left panel: Handwriting `SmoothGradCAMpp` heatmap overlay
- Right panel: Gait feature importance (SHAP bar chart)
- Table below: Top 3–5 features from each modality with their contribution scores

**Grouped SHAP on Fused Vector**
- If early fusion, segment the concatenated vector into handwriting block and gait block
- Use grouped SHAP to compare modality-level contributions

**Counterfactual Analysis (Optional)**
- Minimal perturbation in gait features to flip prediction
- Minimal patch modification in handwriting to flip prediction
- Shows decision boundaries and robustness

---

### Handwriting + Speech

**Dual Saliency Panels**
- Top: Handwriting `SmoothGradCAMpp` overlay on spiral image
- Bottom: Speech `SmoothGradCAMpp` overlay on mel-spectrogram
- Aligned on the same individual for easy visual comparison

**Late Fusion Attention Weights (if applicable)**
- If you implement late fusion with learned attention, visualize the attention weights
- Acts as a natural modality attribution: hand drawing 60%, speech 40%, etc.

**Per-Modality Confidence**
- Show what each modality alone would predict
- Compare to the fused prediction to see synergy/conflict

---

## Trimodal Explainability

### Integrated Trimodal XAI Suite

**1. Individual Modality Saliency (Stacked View)**
```
┌─────────────────────────────────────┐
│  Handwriting SmoothGradCAMpp        │  (spatial regions)
│                                     │
├─────────────────────────────────────┤
│  Speech SmoothGradCAMpp             │  (time-frequency regions)
│                                     │
├─────────────────────────────────────┤
│  Gait SHAP Feature Importance       │  (top 5–10 features)
│                                     │
└─────────────────────────────────────┘
```
- One figure per subject showing all three modality explanations

**2. Global SHAP on 70D Fused Vector**
- Apply SHAP TreeExplainer on the final XGBoost trimodal classifier
- Group features by modality using color/legend:
  - Speech PCA-50 features (blue)
  - Gait features (green)
  - Handwriting PCA-2 features (orange)
- Summary plot shows which features are most important across all samples
- Force plot for individual predictions shows how each feature pushes toward HC or PD

**3. Modality Contribution Summary**
- Compute mean absolute SHAP value for each modality block
- Bar chart: "Speech contributes 45%, Gait 35%, Handwriting 20% to the decision"
- Confidence interval or per-subject breakdown to show variability

**4. Embedding Visualizations (t-SNE / UMAP)**
- Generate t-SNE/UMAP of the 70D fused feature space
- Color by true class (HC/PD)
- Overlay misclassified samples as X markers
- Useful for understanding cluster quality and separability

**5. Confusion Matrix + Explainability Heatmap**
- Standard confusion matrix
- For each misclassified sample, attach a feature importance annotation
- Helps debug: "Why was this HC predicted as PD? Speech was borderline, gait was strong PD..."

---

## Implementation Roadmap

### Phase 1: Unimodal XAI (per modality)
- [ ] **Speech**: `SmoothGradCAMpp` notebook in `02_unimodal/01_speech/`
- [ ] **Handwriting**: `SmoothGradCAMpp` + optional Integrated Gradients in `02_unimodal/02_handwriting/`
- [ ] **Gait**: SHAP + optional temporal saliency in `02_unimodal/03_gait/`

### Phase 2: Bimodal XAI
- [ ] Modality ablation script in `03_bimodal/`
- [ ] Grouped SHAP templates in `03_bimodal/` (reuse across all bimodal pairs)
- [ ] Consistency checking notebook

### Phase 3: Trimodal XAI
- [ ] Integrated analysis notebook in `04_trimodal/analysis.ipynb`
- [ ] Global SHAP on 70D fused vector
- [ ] Modality contribution summary bar chart
- [ ] t-SNE/UMAP plots with class/error overlays

### Phase 4: Inference & Dashboard
- [ ] Shared XAI visualization module in `05_inference/xai_visualizations.py`
- [ ] Streamlit dashboard integration (call visualization functions)
- [ ] Single-sample and batch inference with XAI outputs

---

## Tools & Libraries

| Task | Library | Module | Notes |
|------|---------|--------|-------|
| Grad-CAM++ | torchcam | `SmoothGradCAMpp` | More robust than basic Grad-CAM |
| Integrated Gradients | captum | `IntegratedGradients` | Theoretically grounded; slower |
| SHAP | shap | `TreeExplainer`, `DeepExplainer` | Tree models faster; deep models more flexible |
| Embedding viz | scikit-learn, umap | `TSNE`, `UMAP` | t-SNE slower; UMAP faster, preserves global structure |
| Rendering | matplotlib, opencv | — | Heatmap overlays, figure composition |

---

## Medical Context Considerations

1. **Reproducibility**: Always seed random states for t-SNE, UMAP, and SHAP sampling
2. **Consistency**: Use the same target layer / feature extraction across samples for fair comparison
3. **Calibration**: SHAP values should reflect true model confidence, not raw activation magnitude
4. **Validation**: Compare XAI outputs to domain knowledge (e.g., tremor in handwriting, gait asymmetry)
5. **Isolation**: Test each modality independently to ensure no "spurious" modality collaborations

---

## Example: Single Subject Trimodal Report

```
Subject ID: PD_042
True Label: Parkinson's Disease
Prediction: Parkinson (confidence 87%)

MODALITY CONTRIBUTIONS:
  Speech:       42% (moderate-strong evidence)
  Gait:         38% (moderate-strong evidence)
  Handwriting:  20% (weak evidence)

TOP FEATURES:
  1. Gait feature #7 (stride length asymmetry) → +0.35 SHAP
  2. Speech PC1 (pitch variation range) → +0.28 SHAP
  3. Gait feature #2 (cadence) → +0.22 SHAP
  4. Handwriting PC1 (tremor magnitude) → +0.08 SHAP

SALIENCY MAPS:
  [Handwriting spiral with Grad-CAM overlay highlighting tremulous strokes]
  [Mel-spectrogram with Grad-CAM++ overlay highlighting low-frequency jitter]
  [Gait SHAP bar chart showing stride and cadence features]

INTERPRETATION:
  - Speech and gait provide strong complementary evidence for PD
  - Handwriting less discriminative for this subject but aligns with other modalities
  - Abnormal stride length and pitch variation are key indicators
```

---

## References

- Grad-CAM++: `https://arxiv.org/abs/1710.11063`
- Integrated Gradients: `https://arxiv.org/abs/1703.04730`
- SHAP: `https://arxiv.org/abs/1705.07874`
- torchcam: `https://github.com/frgfm/torch-cam`
- captum: `https://captum.ai/`
- shap: `https://shap.readthedocs.io/`
