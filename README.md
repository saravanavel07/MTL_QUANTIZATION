# MTL_quantization

[![status](https://img.shields.io/badge/status-experimental-orange)](https://github.com/saravanavel07/MTL_QUANTIZATION)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](./LICENSE)

This repository contains Jupyter notebooks and resources for applying quantization techniques to Multi-Task Learning (MTL) models (DNN, CNN).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Notebooks](#notebooks)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The project demonstrates how to apply post-training and/or quantization-aware training to multi-task neural networks to reduce model size and inference cost while retaining acceptable accuracy across tasks. The notebooks include experiments, examples, and utility code to prepare data, train models, and evaluate quantized models.

## Features

- Worked examples in Jupyter Notebooks
- Support for common quantization workflows (post-training quantization, quantization-aware training)
- Examples using DNN and CNN architectures for multi-task learning

## Requirements

Minimum recommended:

- Python 3.8+
- Jupyter Notebook or JupyterLab

A typical set of Python packages used by the notebooks:

- numpy
- pandas
- scikit-learn
- torch (PyTorch)
- torchvision (if using vision models)
- matplotlib
- seaborn
- tqdm
- jupyter

If some notebooks require TensorFlow, add `tensorflow`.

Install with pip:

```bash
pip install -r requirements.txt
```

## Quickstart

1. Clone the repository:

```bash
git clone https://github.com/saravanavel07/MTL_QUANTIZATION.git
cd MTL_QUANTIZATION
```

2. Checkout the docs branch (if using the branch with these docs):

```bash
git checkout docs/readme-requirements
```

3. Install dependencies and run Jupyter:

```bash
pip install -r requirements.txt
jupyter notebook
```

Open the notebooks in your browser and run the cells in order.

## Notebooks

The repository is notebook-first. Look for a `notebooks/` or top-level `.ipynb` files. Recommended starting notebook(s):

- `notebooks/0_data_and_preprocessing.ipynb` (data preparation)
- `notebooks/1_baseline_training.ipynb` (training a floating-point baseline)
- `notebooks/2_quantization.ipynb` (applying quantization and evaluation)

If your repo uses different filenames, open the notebooks you have and follow their first cells for dependencies and instructions.

## Usage

Typical workflow in the notebooks:

1. Data preparation and preprocessing for multi-task datasets
2. Define a multi-task model (shared backbone, task-specific heads)
3. Train a baseline floating-point model
4. Apply post-training quantization or quantization-aware training
5. Evaluate task metrics and model size/performance tradeoffs

Tweak hyperparameters and quantization settings directly in the notebooks to experiment.

## Examples

If you add example outputs (plots, saved models), put them in an `examples/` directory and reference them from the notebooks.

## Contributing

Contributions are welcome. Please open issues for bugs or feature requests and create pull requests for changes. When contributing:

- Add or update notebooks demonstrating new ideas
- Include a brief description and expected results for reproducibility
- Add required package versions to `requirements.txt` if needed

## License

This repository does not contain a LICENSE file yet. If you want an explicit license (recommended), add a `LICENSE` file — e.g., MIT.

## Contact

For questions or help, open an issue in the repository or contact the maintainer.
