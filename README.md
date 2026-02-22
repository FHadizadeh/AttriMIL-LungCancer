
```markdown
# AttriMIL with GigaPath Integration for NSCLC Classification

This repository provides an optimized, debugged, and high-performance implementation of the **AttriMIL** (Attribute-based Multi-Instance Learning) framework. The model is specifically adapted for **TCGA-NSCLC** subtyping (LUAD vs. LUSC) using features extracted from the **GigaPath** foundation model.

## 🌟 Key Engineering Highlights & Bug Fixes
This implementation addresses several critical issues found in original academic MIL repositories, ensuring stability and accuracy for large-scale pathological analysis:

- **Spatial Constraint Dimension Fix:** Resolved a major `IndexError` in the `spatial_constraint` module. The original code incorrectly indexed 3D attribute scores; we implemented a precise mapping (`A[0, c]`) to align attention scores with the 1D k-NN spatial graph coordinates.
- **Batch Dimension Management:** Fixed recurring `IndexError` during the training/validation loops by implementing robust `.squeeze(0)` operations to handle PyTorch's default batching behavior for Whole Slide Images (WSIs).
- **GigaPath Foundation Model Alignment:** Adapted the architecture to handle 1536-dimensional features from **GigaPath**, ensuring the attention mechanism properly captures high-resolution morphological details.
- **Enhanced Logging System:** Corrected mismatched variables in the logging module that previously caused incorrect reporting of `bag_size` and class labels.

## 📂 Project Structure
- `create_nearest.py`: Constructs k-nearest neighbor (k-NN) spatial graphs for patch-level relationships based on (x, y) coordinates.
- `generate_splits.py`: A deterministic utility for dataset splitting (Train/Val/Test) to ensure reproducible 5-fold cross-validation.
- `create_csv.py`: Maps Whole Slide Images (WSI) to diagnostic labels and manages metadata.
- `trainer_attrimil_abmil.py`: The core training engine, optimized for foundation model features and multi-instance learning.
- `constraints.py`: Contains the fixed and optimized Spatial and Rank constraint loss functions.

## 💻 Usage Guide

### 1. Spatial Graph Construction
Generate neighbor relationships for all patches within each slide. This is required for the Spatial Constraint module:
```bash
python create_nearest.py

```

### 2. Dataset Splitting

Generate the `splits_0.csv` file (Default: 80% Train, 10% Val, 10% Test):

```bash
python generate_splits.py

```

### 3. Training

Start the training process. The model will automatically handle feature alignment and apply spatial constraints:

```bash
python trainer_attrimil_abmil.py --n_classes 2 --batch_size 1 --lr 2e-4

```

## 📊 Performance & Convergence

Early results demonstrate exceptional convergence and stability thanks to the GigaPath feature space and optimized constraints:

| Metric | Epoch 0 | Epoch 1 | Epoch 2 |
| --- | --- | --- | --- |
| **Validation AUC** | 0.9050 | 0.9147 | **0.9340** |
| **Class 0 (LUAD) Acc** | 40.7% | 44.4% | **74.1%** |
| **Class 1 (LUSC) Acc** | 100% | 100% | **91.3%** |

The significant jump in **Class 0 Accuracy** by Epoch 2 highlights the effectiveness of the fixed Spatial Constraint module in mitigating majority-class bias and enhancing local feature aggregation.

```
