# 🔬 AttriMIL with Foundation Model Integration for NSCLC Classification

This repository provides an optimized, debugged, and fully evaluated implementation of the **AttriMIL** framework for **TCGA-NSCLC** classification. By replacing traditional CNN backbones with advanced Foundation Models (**UNI / GigaPath**), we establish a new state-of-the-art benchmark for explainable digital pathology.

## 💡 Problem Statement & Innovation

While the original AttriMIL framework demonstrated strong performance, its official implementation was heavily coupled with traditional CNN backbones and limited interpretability tools.

**Our Innovation:** We engineered a seamless integration pipeline to leverage **UNI/GigaPath** embeddings ($D=1536$). This allows AttriMIL to utilize highly compositional and semantic representations, significantly boosting diagnostic accuracy. Furthermore, we've automated the link between abstract model scores and specific histological traits (e.g., Necrosis, Keratinization).

---

## ⚙️ Feature Alignment & Structural Integration

The core technical challenge was bridging the gap between high-capacity Foundation Models and the MIL architecture.

### 1. Dimensionality Alignment Layer

Raw features from UNI/GigaPath are extracted as high-dimensional vectors ($D=1536$). We implemented a **Trainable Linear Projection Bottleneck** to map these into the 512-d latent space required by AttriMIL. This layer is crucial for:

* **Aligning semantic embeddings** with the Multi-Branch Attribute Scoring (MBAS) mechanism.
* **Reducing computational overhead** while preserving the "compositional" knowledge of the foundation model.

### 2. Topological Alignment

Using the `ball_tree` algorithm, we reconstructed the spatial $k$-NN graph ($k=9$) to align visual features with their original physical coordinates in the gigapixel WSI.

---

## 🛠️ Data Processing & Harmonization Pipeline

1. **Data Acquisition:** Secured official access to TCGA-NSCLC and utilized pre-extracted UNI features to bypass redundant local WSI pre-processing.
2. **GigaPath Alignment:** Adapted the architecture to handle 1536-dimensional features from the GigaPath vision transformer.
3. **Data Harmonization:** Developed `create_nearest.py` to standardize raw `.h5` files into a unified format compatible with AttriMIL's dataloader.
4. **Dynamic Label Mapping:** Custom `create_labels.py` script for safe, leak-free parsing of massive TCGA directories.
5. **Pre-computed Master Database:** Implemented a pre-calculation stage (`precompute_patches.py`) that stores top/bottom attribute scores for all slides, reducing concept visualization time from minutes to seconds.

---

## 🌟 Key Engineering Highlights & Bug Fixes

* **Model Output Alignment:** Fixed `ValueError: too many values to unpack` by correctly mapping the 5-tuple outputs from the modified forward pass.
* **Spatial Constraint Dimension Fix:** Resolved a major `IndexError` in the `spatial_constraint` module by implementing precise mapping ($A[0, c]$).
* **Batch Dimension Management:** Fixed recurring `IndexError` issues using robust `.squeeze(0)` operations for foundation model tensors.
* **Early Stopping Optimization:** Refactored the mechanism to rely on a **15-epoch patience threshold**, preventing overfitting and saving GPU resources.

---

## 📊 Final Performance & Baseline Comparison

Evaluated on **500 unseen slides** (250 LUAD, 250 LUSC) using 1000-iteration Monte Carlo bootstrapping.

| Backbone | Method Category | Method | AUC-ROC | F1-Score | Accuracy |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 (SimCLR) | Global Attention | TransMIL | 0.948 ± 0.011 | 0.878 ± 0.033 | 0.890 ± 0.027 |
| ResNet-18 (SimCLR) | AttriMIL Variants | TransMIL + AttriMIL | 0.959 ± 0.014 | 0.906 ± 0.032 | 0.911 ± 0.024 |
| **Foundation (Ours)** | **Integrated Pipeline** | **ABMIL + AttriMIL** | **0.9781 ± 0.006** | **0.9323 ± 0.011** | **0.9324 ± 0.011** |

---

## 🎨 Visual Interpretability & Morphological Validation

### 1. Spatial Attribute Heatmaps

The model generates continuous scoring maps highlighting diagnostic regions. High-score regions (Red) indicate strong morphological evidence for the predicted class.
*(File: `evaluation_results/heatmap.png`)*

### 2. Attribute-Specific Patch Analysis & Interpretations

We utilize the **MBAS module** to extract representative patches for specific clinical concepts.

* **A. Necrosis (Tissue Death):** * *Interpretation:* TOP patches localize to regions of **eosinophilic, amorphous debris** and nuclear fragmentation (karyorrhexis). BOTTOM patches show viable tumor clusters.
* *Visual:* `evaluation_results/visualizations/necrosis_vis.png`


* **B. Keratinization (Squamous Marker):** * *Interpretation:* TOP patches focus on **keratin pearls** and dense, eosinophilic cytoplasm (LUSC markers). BOTTOM patches isolate irrelevant alveolar spaces.
* *Visual:* `evaluation_results/visualizations/keratinization_vis.png`


* **C. Differentiation (Architectural Maturity):** * *Interpretation:* TOP patches highlight **well-organized glandular structures** with distinct lumens (LUAD markers). BOTTOM patches capture anaplastic growth patterns.
* *Visual:* `evaluation_results/visualizations/differentiation_vis.png`



---

## 📂 Project Structure

* `create_nearest.py`: Harmonizes features and constructs $k$-NN spatial graphs.
* `precompute_patches.py`: Generates a global attribute database for zero-latency visualization.
* `concept_extractor.py`: CLI tool for automated concept extraction, GDC downloading, and rendering.
* `trainer_attrimil_abmil.py`: Core engine with fixed spatial constraints and optimized early-stopping.
* `bootstrap_evaluation.py`: statistical engine for precise 95% confidence intervals.
* `evaluation_results/`: Central directory for metrics, heatmaps, and visual galleries.

---

## 💻 Usage Guide

1. **Spatial Graph Construction:** `python create_nearest.py`
2. **Dataset Splitting:** `python generate_splits.py`
3. **Training:** `python trainer_attrimil_abmil.py --batch_size 1 --lr 2e-4`
4. **Pre-compute Attributes:** `python precompute_patches.py --h5_dir "./h5_features"`
5. **Automated Concept Extraction:** `python concept_extractor.py --concept "necrosis" --auto_download`
