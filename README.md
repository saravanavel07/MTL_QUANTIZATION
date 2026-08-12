# MTL_quantization

This repository contains Jupyter notebooks and resources for applying quantization techniques to Multi-Task Learning (MTL) models (DNN, CNN).

## Overview

The project demonstrates how to apply post-training and/or training-aware quantization to multi-task neural networks to reduce model size and inference cost while retaining acceptable accuracy across tasks. The notebooks include experiments, examples, and utility code to prepare data, train models, and evaluate quantized models.

## Features

- Worked examples in Jupyter Notebooks
- Support for common quantization workflows (post-training quantization, quantization-aware training)
- Examples using DNN and CNN architectures for multi-task learning

## Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Common ML libraries (examples below)

A typical set of Python packages required:

- numpy
- pandas
- scikit-learn
- torch (PyTorch)
- torchvision (if using vision models)
- tensorflow (if notebooks target TF; check individual notebooks)
- matplotlib

You can install requirements with pip (adjust packages to match your notebooks):

pip install -r requirements.txt

If there is no requirements.txt in this repository, install the packages you need manually, for example:

pip install numpy pandas scikit-learn torch torchvision matplotlib jupyter

## Notebooks

Open the Jupyter notebooks in this repository to explore the experiments. Typical workflows in the notebooks:

1. Data preparation and preprocessing for multi-task datasets
2. Defining a multi-task model (shared backbone, task-specific heads)
3. Training a baseline floating-point model
4. Applying post-training quantization or quantization-aware training
5. Evaluating task metrics and model size/performance tradeoffs

## Usage

1. Clone the repository:

git clone https://github.com/saravanavel07/MTL_QUANTIZATION.git
cd MTL_QUANTIZATION

2. Install dependencies (see Requirements)
3. Start Jupyter and open the notebooks:

jupyter notebook

4. Run the cells in order. Modify hyperparameters and quantization settings in the notebooks to experiment.

## Examples

See the notebooks for concrete examples. If you want to add example outputs or artifacts (trained models, plots), place them in a new `examples/` directory and reference them in the notebooks.

## Contributing

Contributions are welcome. Please open issues for bugs or feature requests and create pull requests for changes. When contributing:

- Add or update notebooks demonstrating new ideas
- Include a brief description and expected results for reproducibility
- Add required package versions to `requirements.txt` if needed

## License

If you want a license, add a `LICENSE` file to the repository. If none is provided, include one (e.g., MIT) to make reuse clear.

## Contact

For questions or help, open an issue in the repository or contact the maintainer.
