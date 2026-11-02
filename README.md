# MTL Quantization

A research-oriented PyTorch project for exploring quantization and sparsity in Multi-Task Learning (MTL). The repository combines a shared ResNet-style backbone with multiple task-specific heads for NYU-v2 based scene understanding tasks, with experiments focused on monitoring compression behavior and multi-task performance trade-offs.

## Overview

This project investigates how quantization-aware training and sparsity analysis affect a multi-task model that jointly predicts:

- semantic segmentation
- surface normals
- depth estimation

The implementation centers around a shared feature extractor and task-specific output heads, with training and validation pipelines written in PyTorch.

## Motivation

Multi-task learning aims to train one model that can solve several related problems at once while sharing representational capacity. In this repository, the emphasis is on understanding whether compression and sparsity techniques can reduce memory and computation without significantly harming task performance.

## Repository Structure

```text
MTL_QUANTIZATION/
├── MTL_test1.ipynb                 # Notebook-based exploratory workflow
├── MTL_test2.py                   # Main training and evaluation pipeline
├── TreeMTL/                       # Multi-task model and dataset utilities
├── DGMSParent/                    # DGMS-related utilities and model components
├── experiment_results/            # Experiment result artifacts
├── graphs/                         # Generated plots and visual summaries
├── __init__.py                    # Package marker
├── actual_jobs.txt                # Job/task metadata
├── check_sparsity.py              # Sparsity analysis utilities
├── compression_rate.txt           # Compression and sparsity metrics
├── error_bar.py                   # Visualization helpers for error bars
├── individual_model_evals.txt     # Per-model evaluation data
├── model.txt                      # Serialized model description
├── notes.txt                      # Notes and observations
├── output.txt                     # Training output logs
├── output1.txt                    # Additional output logs
├── output2.txt                    # Additional output logs
├── pretrained_features.txt        # Pretrained feature info
├── print_size_of_model.py         # Size and parameter summary utility
├── requirements.txt               # Python dependency list
├── validation_results.txt         # Validation metrics output
├── validation_results2.txt        # Additional validation metrics output
├── val30kiters.txt                # Validation run data
├── val600iters.txt                # Validation run data
├── val600iters_savedmodels.txt    # Saved model metadata
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules (if present locally)
```

## Core Components

### 1. Training Pipeline

The main pipeline is implemented in `MTL_test2.py` and includes:

- dataset loading for NYU-v2
- model construction with a shared backbone and task heads
- multi-task loss aggregation
- training and validation loops
- logging and result saving

### 2. Multi-Task Model

The model design is centered on a shared backbone with independent output heads for each task. The task set in the project is:

```python
tasks = ('segment_semantic', 'normal', 'depth_zbuffer')
```

### 3. Quantization and Sparsity Analysis

The project includes utilities for:

- measuring sparsity
- evaluating compression trends
- visualizing validation behavior
- comparing model performance under different experiment settings

## Requirements

This project targets a Python environment with PyTorch and related scientific libraries installed.

### Python

- Python 3.8+

### Dependencies

```bash
pip install -r requirements.txt
```

The repository's dependency list includes:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyter
- torch
- torchvision
- tqdm

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/saravanavel07/MTL_QUANTIZATION.git
cd MTL_QUANTIZATION
```

### Create a Virtual Environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Setup

The project uses NYU-v2 data and references a local dataset directory in the training script:

```python
dataroot = "/home/sbajaj/MyCode/quantization_remote/Datasets/nyu_v2"
```

Before running the project, update this path to match your local environment or your dataset location.

## Running the Project

The primary experiment script is:

```bash
python MTL_test2.py
```

This script typically performs the following:

1. loads the NYU-v2 dataset
2. instantiates the multi-task model
3. builds the loss and metric dictionaries for each task
4. trains the model with a shared backbone and task heads
5. logs validation results
6. saves experiment artifacts and output summaries

## Output Files

The repository generates several result files during training and evaluation, including:

- `validation_results.txt`
- `validation_results2.txt`
- `compression_rate.txt`
- `actual_jobs.txt`
- `individual_model_evals.txt`
- `output*.txt`
- `graphs/`

These files help track model quality, task metrics, and compression-related measurements.

## Notes and Limitations

This project is research-focused and includes environment-specific paths and experimental artifacts. Some files appear to be tied to local execution setups, so users may need to:

- adjust dataset paths
- update dependency versions for their environment
- verify submodule or package availability for custom utilities
- run notebook experiments separately if needed

## Suggested Improvements

For a cleaner and more reproducible research workflow, the project could benefit from:

- a configuration file for hyperparameters
- standardized output directories
- explicit README setup instructions for dataset acquisition
- version pinning for PyTorch and related libraries
- optional command-line arguments for training configuration
- a license file if the project is intended for broader distribution

## Research Use

This repository is suitable for:

- multi-task learning research
- neural network compression experiments
- sparsity-aware model evaluation
- exploring quantization trade-offs in dense prediction tasks
- analyzing multi-task performance under shared representation constraints

## Contributing

Contributions, issue reports, and improvements are welcome. If you use or extend this project, consider documenting:

- dataset preparation steps
- reproduction commands
- model configuration changes
- metric interpretation

## License

This repository does not currently appear to include a license file. If you intend to distribute or share the project publicly, consider adding an appropriate open-source license.

## Summary

`MTL_QUANTIZATION` is a compact but research-oriented PyTorch repository for studying multi-task learning under compression-oriented constraints. It is a good fit for experimentation, model analysis, and exploration of quantization effects in a shared-backbone multi-task setting.
