# Facial Emotion Recognition (FER) with a Shared CNN

## Authors
- Kareem Atout (GitHub: @kareematout)
- Muhammad Kamran Khan (GitHub: @kamivibra111)

---

## Description of Question and Research Topic:
Facial expressions convey non-verbal signals commonly grouped into seven basic emotions: angry, disgust, fear, happy, sad, surprise, and neutral. This project studies whether a single, compact convolutional neural network (CNN) trained from scratch can accurately classify emotions across two different datasets. We evaluate accuracy, macro-F1, and per-class precision/recall, and test robustness to blur, occlusion, and low light to understand real-world performance. Gradient-based saliency visualizations are used to inspect which facial regions influence predictions and to analyze common failure modes (e.g., fear vs. surprise). We further perform cross-dataset experiments (train on Dataset-A, test on Dataset-B and vice versa) to quantify domain shift and validate the generality of the shared CNN.

---

## Project Outline / Plan
1) Data acquisition & preprocessing
   - Two datasets (one per partner), standardized to the same 7-class label schema.
   - Shared preprocessing pipeline (resize, normalization, augmentations).

2) Modeling (CNNs)
   - Define one shared CNN architecture and a single training recipe.
   - Run two experiments with the same model: Exp-A on Dataset-A and Exp-B on Dataset-B.

3) Evaluation & analysis
   - Metrics: accuracy, macro/micro-F1, per-class precision/recall; confusion matrices.
   - Robustness stress tests: performance deltas under blur/occlusion/low-light.
   - plot losses/accuracy/validation, show validation images with model's predicted outcome

4) Reproducibility & release
   - Deterministic seeds, requirements.txt, data access notes, scripts to reproduce preprocessing/training/evaluation, and a Model Card.

---

## Notebooks & Authorship (4 total)
- Notebook 1 — Data Pre-processing (Kareem Atout):
  Dataset-A download, organization, splits, transforms/augmentations, integrity checks, class balance plots.
- Notebook 2 — Data Pre-processing (Muhammad Kamran Khan):
  Dataset-B download, organization, splits, transforms/augmentations, integrity checks, class balance plots.
- Notebook 3 — Model Construction (Team):
  Defines one shared CNN and training loop; runs Exp-A (Dataset-A) and Exp-B (Dataset-B); logs metrics; saves checkpoints.
- Notebook 4 — Analysis & Visualization (Co-authored):
  Confusion matrices, robustness results, plotting of results, and discussion.

---

## Data Collection Plan (two parts)

### Muhammad Kamran Khan — Dataset-A (FER2013)
- Access: Public academic dataset.
- Preparation:
  - Convert to image folders and map labels to {angry, disgust, fear, happy, sad, surprise, neutral}.
  - Resize to 64×64 or 96×96 (grayscale or 3-channel), normalize; create train/val/test splits.
  - Augmentations: random horizontal flip, small rotation, brightness/contrast jitter, Gaussian blur, random occlusion patch.

### Kareem Atout — Dataset-B (RAF-DB Basic 7-class)
- Access: Download per dataset terms to data/rafdb/raw/.
- Preparation:
  - Face crop/alignment if provided; resize/normalize identically to Dataset-A; create train/val/test splits.
  - Augmentations: mirror Dataset-A’s policy for fair robustness comparison.

Data Access Statement:
This repository does not include data files. Expected layout after prep:
    ```text
    data/
      fer2013/
        train/
        val/
        test/
      rafdb/
        train/
        val/
        test/
    ```
    
---

## Model Plans (two parts; shared architecture)

### Kareem Atout — Model Plan for Dataset-A
- Architecture: 4 convolutional blocks (Conv 3×3 → BatchNorm → ReLU → MaxPool) ×4 → GlobalAveragePool → Dropout → Linear(7).
- Training: Adam optimizer, learning-rate, early stopping on validation macro-F1; class weighting or focal loss if needed.
- Experiment A: Train and evaluate on Dataset-A; save best checkpoint and metrics.

### Muhammad Kamran Khan — Model Plan for Dataset-B
- Architecture: Identical to Dataset-A mdel plan.
- Training: Same recipe; small hyperparameter sweep (learning rate, dropout) if dataset size differs.
- Experiment B: Train and evaluate on Dataset-B; save best checkpoint and metrics.

---

## Analysis & Visualization (shared)
- Confusion matrices per dataset; per-class precision/recall; macro/micro-F1.
- Robustness curves: metric deltas under blur/occlusion/low-light perturbations.
- Plotting and visualizting: show and plot results including accuracy and losses
- Error analysis: gallery of frequent confusions (e.g., fear ↔ surprise).

---

## Project Timeline
- Week 1:
  Finalize topic/scope; create repo; install environment; download datasets; complete both pre-processing notebooks.
- Week 2:
  Implement shared CNN; run Exp-A and Exp-B; record baseline metrics.
- Week 3:
  Hyperparameter tuning; robustness experiments; saliency; draft Analysis notebook.
- Week 4:
  Finalize Analysis notebook; polish figures; prepare slides.
- Presentation:
  8 minutes total (~4 minutes per person) during Week 16.
- Final Components Due:
  Per syllabus (ensure GitHub quality, reproducibility, and Model Card are complete).

---

## Reproducibility

Environment:
Python 3.10+, PyTorch, torchvision, numpy, pandas, scikit-learn, matplotlib.

Determinism:
Set seeds for random, numpy, and torch.
