# MTL Quantization for Multi-Task Learning

A research project exploring model compression in **multi-task dense prediction** with PyTorch. The experiments combine a shared ResNet-18 feature extractor with task-specific prediction heads and investigate quantization, sparsity, model size, and inference performance on the NYU-v2 dataset.

## Project at a glance

The model is configured to predict three tasks from the same input:

- **Semantic segmentation** (`segment_semantic`, 40 classes)
- **Surface-normal estimation** (`normal`, 3 channels)
- **Depth estimation** (`depth_zbuffer`, 1 channel)

The shared representation is produced by a pretrained ResNet-18 backbone. Each task has its own ASPP-based output head. Training and validation use task-specific losses and metrics.

## Components and technologies

| Component | Purpose in this project |
|---|---|
| Python and Jupyter Notebook | Experiment scripts and interactive exploration (`MTL_test2.py`, `MTL_test1.ipynb`) |
| PyTorch | Neural-network modules, training, optimization, checkpointing, and tensor operations |
| Torchvision | ResNet-18 backbone and image utilities |
| NYU-v2 dataset utilities (`TreeMTL`) | Dataset loading, task losses, task metrics, and ASPP task heads |
| DGMS components (`DGMSParent`) | Custom convolution modules and sparsity-related utilities used in the DGMS experiments |
| PyTorch quantization | Eager-mode quantization utilities in `print_size_of_model.py`, including quant/dequant stubs and model size comparisons |
| NumPy and tqdm | Numeric processing and progress reporting |
| Matplotlib and Seaborn | Plotting and experiment visualization |
| TensorBoard | Training summaries and histogram logging via PyTorch's TensorBoard integration |

The project explores two related compression workflows: custom `DGMSConv` layer conversion and sparsity analysis, and PyTorch's built-in quantization flow in the model-size evaluation script. These workflows are experimental and are not necessarily enabled by default.

## Repository structure

```text
MTL_QUANTIZATION/
├── MTL_test1.ipynb              # Notebook experiments
├── MTL_test2.py                 # Multi-task model training and validation
├── TreeMTL/                     # Dataset, model-head, loss, and metric utilities
├── testing_pytorch_quantiation/ # PyTorch quantization experiments
├── experiment_results/          # Experiment artifacts
├── graphs/                      # Plots and visualizations
├── check_sparsity.py            # Model sparsity analysis
├── print_size_of_model.py       # Quantized model size and inference evaluation
├── error_bar.py                 # Error-bar and experiment plots
├── requirements.txt             # Python dependencies
└── *.txt                        # Experiment logs, metrics, and notes
```

Some scripts import `DGMSParent` modules. Make sure that package is available in your environment and on `PYTHONPATH`; it may need to be obtained or configured separately if it is not present in your checkout.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/saravanavel07/MTL_QUANTIZATION.git
cd MTL_QUANTIZATION
```

### 2. Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dependency file includes NumPy, pandas, scikit-learn, Matplotlib, Seaborn, Jupyter, PyTorch, Torchvision, and tqdm. Install a PyTorch build compatible with your operating system and CUDA setup if you plan to use a GPU. TensorBoard support may also require installing `tensorboard` separately in your environment.

## Data and configuration

The training and evaluation scripts expect the **NYU-v2** dataset and currently contain machine-specific dataset and checkpoint paths. Before running them:

1. Obtain and prepare NYU-v2 in the format expected by `TreeMTL.data.nyuv2_dataloader_adashare.NYU_v2`.
2. Change the `dataroot` and checkpoint/output paths in the script you intend to run to match your local setup.
3. Ensure `TreeMTL` and the required `DGMSParent` modules can be imported.

The checked-in scripts include absolute paths from the original experiment environment, so they will not work unchanged on most machines. `MTL_test2.py` also calls `.cuda()` directly; as written, it expects a CUDA-enabled environment.

## Running experiments

The primary training script is:

```bash
python MTL_test2.py
```

In the current script, `perform_quantization` is set to `False` by default. Review the configuration near the top of the file before running and enable the appropriate workflow only after verifying the custom dependencies, dataset, paths, and hardware setup.

Other entry points include:

- `jupyter notebook MTL_test1.ipynb` — inspect and run notebook experiments.
- `python check_sparsity.py` — analyze sparsity of a configured/saved model; update its dataset and checkpoint paths first.
- `python print_size_of_model.py` — compare model sizes and evaluate inference behavior using the quantization utilities; update its imports and paths first.

## What the experiments measure

The repository includes code and result files for examining:

- Per-task training and validation losses and metrics
- Weight sparsity and compression-related measurements
- Model checkpoint size before and after quantization
- Inference timing and potential speedup
- Experiment trends visualized with plots and error bars

Results depend on the dataset preparation, model/checkpoint, quantization configuration, device, and software versions. Treat saved text files and plots as experiment artifacts rather than guaranteed benchmark results.

## Reproducibility notes

- Dependency versions are not pinned in `requirements.txt`; record your Python, PyTorch, Torchvision, CUDA, and TensorBoard versions when reproducing an experiment.
- Set and document random seeds and experiment parameters when comparing runs.
- Confirm that the NYU-v2 data paths and custom `TreeMTL`/`DGMSParent` modules match the code's expectations.
- Check the script configuration carefully: several files are research scripts tied to a specific environment, rather than a fully configurable command-line application.

## License

No license file is currently provided. Unless a license is added, reuse and redistribution are not explicitly granted.
