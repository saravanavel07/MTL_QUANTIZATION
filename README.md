# MTL Quantization

A research-focused PyTorch repository for exploring quantization in Multi-Task Learning (MTL), using a shared backbone and multiple task heads for semantic segmentation, surface normals, and depth estimation.

This project is built around experiments on the NYU-v2 dataset and integrates a ResNet-based multi-task architecture with quantization-oriented sparsity analysis.

## Overview

The repository demonstrates how to:

- train a multi-task network with shared feature extraction,
- evaluate several tasks jointly,
- compare standard vs. quantized settings,
- monitor validation metrics and compression-related behavior,
- log experiment outputs for analysis.

The main training pipeline is implemented in `MTL_test2.py`, while `MTL_test1.ipynb` provides a notebook-based exploration workflow.

## Key idea

Multi-task learning enables a single model to solve multiple objectives at once. In this project, the model is trained for:

- `segment_semantic`
- `normal`
- `depth_zbuffer`

The goal is to evaluate whether compression and sparsity techniques can reduce resource usage without destroying task performance.

## Repository structure

```text
MTL_QUANTIZATION/
├── MTL_test1.ipynb                # Notebook-based experimentation
├── MTL_test2.py                  # Main training and evaluation pipeline
├── TreeMTL/                     # Multi-task model/data utilities
├── DGMSParent/                  # DGMS-related model utilities (if present in local env)
├── experiment_results/           # Result artifacts and experiment summaries
├── graphs/                      # Visualization outputs
├── check_sparsity.py             # Sparsity analysis utilities
├── error_bar.py                 # Error-bar plotting helpers
├── print_size_of_model.py        # Model size/parameter print utility
├── requirements.txt              # Python dependencies
├── validation_results.txt        # Validation logs
├── compression_rate.txt          # Compression metrics
├── actual_jobs.txt               # Job/task metadata
├── output*.txt                   # Saved output logs
├── notes.txt                     # Notes and observations
├── model.txt                    # Serialized model summary
├── README.md                    # Project documentation
└── __init__.py
```

## Core workflow

The project follows this general pipeline:

1. Load NYU-v2 data for training and validation.
2. Create a multi-task model with a shared backbone and task-specific heads.
3. Train using composite multi-task loss.
4. Validate performance on each task.
5. Track sparsity/compression statistics.
6. Save result summaries and checkpoint artifacts.

## Model configuration

The model uses a ResNet-style backbone with shared latent features and task-specific output heads. In the current implementation, the backbone is based on `torchvision.models.resnet18` and is adapted for the multi-task setting.

The tasks and classes are defined as follows:

```python
tasks = ('segment_semantic', 'normal', 'depth_zbuffer')

clsNum = {
    'segment_semantic': 40,
    'normal': 3,
    'depth_zbuffer': 1,
}
```

This setup is useful for multi-objective experiments where a single model must produce multiple outputs from the same input image.

## Installation

Clone the repository:

```bash
git clone https://github.com/saravanavel07/MTL_QUANTIZATION.git
cd MTL_QUANTIZATION
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Typical environment requirements include:

- Python 3.8+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter
- tqdm

## Running the project

The main experiment script is:

```bash
python MTL_test2.py
```

This script initializes the data loaders, builds the model, runs training and validation loops, logs metrics, and saves output files.

## Experiment outputs

The project produces several result files, including:

- `validation_results.txt` for validation metrics
- `compression_rate.txt` for sparsity/compression statistics
- `actual_jobs.txt` and `individual_model_evals.txt` for job and evaluation traces
- `graphs/` for generated plots and visual summaries
- `output*.txt` for logs generated during training runs

## Dependencies and notes

This repository is strongly tied to research code and experimental workflows. Some paths and local dataset references are environment-specific, such as:

```python
dataroot = "/home/sbajaj/MyCode/quantization_remote/Datasets/nyu_v2"
```

When running locally, update dataset paths to match your filesystem.

## Use cases

This codebase is suitable for:

- multi-task learning research,
- neural network compression experiments,
- sparsity-aware quantization analysis,
- validation of task trade-offs under model compression,
- visualization of training and evaluation trends.

## Potential next steps

Possible improvements for this repository include:

- adding a clean project-level configuration file,
- creating a reproducible notebook walkthrough,
- standardizing output directories,
- documenting dataset setup and task definitions,
- adding training commands and evaluation examples for reproducibility,
- adding a `LICENSE` file if this project is shared publicly.

## Summary

`MTL_QUANTIZATION` is a compact but research-oriented project for experimenting with model compression in multi-task learning. It combines PyTorch training, NYU-v2 data handling, multi-task optimization, and quantization/sparsity analysis to study how efficient models behave under shared representations.

## Contact

For questions or collaboration, use the repository issue tracker on GitHub.
