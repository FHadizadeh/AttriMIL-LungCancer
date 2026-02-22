# AttriMIL with GigaPath Integration for NSCLC Classification

This repository provides an optimized and debugged implementation of the **AttriMIL** framework for **TCGA-NSCLC** classification using **GigaPath** features.

## 🌟 Key Engineering Highlights & Bug Fixes

This implementation addresses several critical issues found in original academic MIL repositories to ensure stability and accuracy:

- **Spatial Constraint Dimension Fix:** Resolved a major `IndexError` in the `spatial_constraint` module. We implemented a precise mapping (`A[0, c]`) to align attention scores with the 1D k-NN spatial graph coordinates.
- **Batch Dimension Management:** Fixed recurring `IndexError` during training/validation loops by implementing robust `.squeeze(0)` operations.
- **GigaPath Foundation Model Alignment:** Adapted the architecture to handle 1536-dimensional features from the GigaPath vision transformer.
- **Enhanced Logging System:** Corrected mismatched variables that previously caused incorrect reporting of `bag_size` and class labels.

## 📂 Project Structure

- `create_nearest.py`: Constructs k-nearest neighbor (k-NN) spatial graphs for patch-level relationships.
- `generate_splits.py`: A deterministic utility for 80-10-10 dataset splitting.
- `create_csv.py`: Maps Whole Slide Images (WSI) to diagnostic labels.
- `trainer_attrimil_abmil.py`: The core training engine.
- `constraints.py`: Contains the fixed Spatial and Rank constraint loss functions.

## 💻 Usage Guide

### 1. Spatial Graph Construction
Generate neighbor relationships for all patches:
```bash
python create_nearest.py

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
