# AttriMIL with Foundation Model Integration for NSCLC Classification

This repository provides an optimized, debugged, and fully evaluated implementation of the **AttriMIL** framework for **TCGA-NSCLC** classification (Lung Adenocarcinoma vs. Squamous Cell Carcinoma) using Foundation Model features (UNI / GigaPath).

## 🌟 Key Engineering Highlights & Bug Fixes

This implementation addresses several critical issues found in original academic MIL repositories to ensure stability, accuracy, and robust evaluation:
* **Spatial Constraint Dimension Fix:** Resolved a major `IndexError` in the `spatial_constraint` module. We implemented a precise mapping (`A[0, c]`) to align attention scores with the 1D k-NN spatial graph coordinates.
* **Batch Dimension Management:** Fixed recurring `IndexError` issues during training/validation loops by implementing robust `.squeeze(0)` operations.
* **Foundation Model Alignment:** Adapted the architecture and unpacking logic to handle high-dimensional features from vision transformers, safely managing complex tuple outputs.
* **Robust Evaluation Pipeline:** Corrected path duplication bugs (`h5_coords_files/h5_coords_files/`) and implemented a dynamic, fuzzy-matching label generator to perfectly map massive TCGA datasets without dropping slides.
* **Vision-Language Clinical Alignment:** Built a custom extraction pipeline to pull continuous maximum attribute scores from the model's internal pathways, enabling statistical correlation with textual diagnostic reports.

## 📂 Project Structure

* `create_nearest.py`: Constructs k-nearest neighbor (k-NN) spatial graphs for patch-level relationships.
* `generate_splits.py`: A deterministic utility for 80-10-10 dataset splitting.
* `create_csv.py`: Maps Whole Slide Images (WSI) to diagnostic labels.
* `trainer_attrimil_abmil.py`: The core training engine featuring fixed Spatial and Rank constraints.
* `tester_attrimil_abmil.py`: The evaluation engine. Calculates final metrics, confusion matrices, and extracts continuous visual attribute scores.
* `evaluation_results/`: Directory containing the final outputs, including `summary.csv`, detailed classification reports, and visual plots.

## 📊 Final Performance on 500 Unseen Test Slides

After resolving early convergence issues and training for 50 epochs, the model was evaluated on a perfectly balanced test set of **500 unseen TCGA slides** (250 LUAD, 250 LUSC). 

The foundation model embeddings paired with the AttriMIL framework achieved exceptional diagnostic capabilities:

| Metric | Score | Note |
| :--- | :--- | :--- |
| **AUC-ROC** | **0.9780** | Near-perfect class separation |
| **Accuracy** | **0.9320** | 466 / 500 correctly classified |
| **F1-Score** | **0.9323** | Highly balanced precision/recall |
| **Precision** | **0.9286** | Positive Predictive Value |
| **Recall** | **0.9360** | Sensitivity / True Positive Rate |
| **Specificity**| **0.9280** | True Negative Rate |

### Diagnostic Visualizations
*(Stored in `evaluation_results/diagnostic_plots.png`)*
![Diagnostic Plots](evaluation_results/diagnostic_plots.png)

## 🧬 Vision-Language Clinical Alignment

Beyond standard classification, this repository includes tools to validate the model's biological interpretability. By extracting the **Maximum Attribute Scores** (`max_attr_LUAD` and `max_attr_LUSC`) from the model's logits, we can calculate the Point-Biserial Correlation against binary clinical keywords (e.g., Necrosis, Keratinization, Mucin) extracted from pathologists' diagnostic reports.

*(Stored in `evaluation_results/clinical_alignment_heatmap.png`)*
![Clinical Alignment Heatmap](evaluation_results/clinical_alignment_heatmap.png)

## 💻 Usage Guide

### 1. Spatial Graph Construction
Generate neighbor relationships for all patches:
```bash
python create_nearest.py

### 2. Dataset Splitting
Generate the splits_0.csv file (Default: 80% Train, 10% Val, 10% Test):
```bash
python generate_splits.py

### 3. Training
Start the training process. The model will automatically handle feature alignment and apply spatial constraints:
```bash
python trainer_attrimil_abmil.py --n_classes 2 --batch_size 1 --lr 2e-4

### 4. Evaluation & Attribute Extraction
Run the testing pipeline to evaluate the saved weights against the test set, generate ROC/Confusion Matrix plots, and extract the continuous visual scores for clinical alignment:
```bash
python tester_attrimil_abmil.py
