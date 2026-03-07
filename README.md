# Deep Learning Assignment 1

NumPy implementation of a fully-connected neural network for MNIST/Fashion-MNIST, with experiment scripts for optimizer, activation, loss-function, initialization, and error-analysis comparisons using Weights & Biases (W&B).

## Project Structure

```text
.
|-- src/
|   |-- ann/                         # Core neural network code (layers, activations, optimizers, losses)
|   |-- train.py                     # Main training script (+ optional sweep mode)
|   |-- run_compare.py               # Optimizer-only sweep/comparison
|   |-- run_activation_compare.py    # ReLU vs Sigmoid gradient-flow analysis
|   |-- lossfunction_comparison.py   # Cross-entropy vs MSE sweep
|   |-- dead_neuron_investigation.py # ReLU dead neurons vs tanh saturation
|   |-- error_analysis.py            # Best-run confusion matrix + failure gallery
|   |-- global_performance.py        # Train vs test accuracy overlay from W&B runs
|   |-- weight_initialization.py     # Zero vs Xavier symmetry/gradient analysis
|   |-- fashion_mnist.py             # Three fixed Fashion-MNIST runs
|   `-- inference.py                 # Inference entrypoint (stub)
|-- notebooks/
|-- sweep_config.yaml
`-- README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install numpy matplotlib scikit-learn pyyaml wandb keras
```

If `keras.datasets` fails in your environment, install TensorFlow as backend:

```bash
pip install tensorflow
```

## Quick Start

Run from the project root:

```bash
python src/train.py --dataset mnist --epochs 10 --optimizer adam --activation_function relu
```

This logs `train/*`, `val/*`, and `test/*` metrics to W&B.

## Main Experiment Commands

### 1) Hyperparameter Sweep (from `sweep_config.yaml`)

```bash
python src/train.py --run_sweep --sweep_config sweep_config.yaml --sweep_count 100
```

### 2) Optimizer Comparison (fixed architecture)

```bash
python src/run_compare.py --optimizers sgd,momentum,nesterov,rmsprop,adam --agent_count 5
```

### 3) Activation Comparison (ReLU vs Sigmoid with Adam)

```bash
python src/run_activation_compare.py --configs 2x128,3x128,5x128 --epochs 10
```

Saves gradient plot to `activation_grad_norms.png` by default.

### 4) Loss Function Comparison (Cross-Entropy vs MSE)

```bash
python src/lossfunction_comparison.py --epochs 10 --run_count 10
```

### 5) Dead Neuron Investigation (ReLU vs tanh)

```bash
python src/dead_neuron_investigation.py --epochs 20 --hidden_layers 3 --hidden_size 128
```

### 6) Error Analysis (best run from W&B)

```bash
python src/error_analysis.py --entity <wandb_entity> --project <wandb_project> --metric_key test/accuracy
```

Outputs are saved in `src/models/` (confusion matrix, top confusion pairs, failure gallery).

### 7) Global Performance Overlay

```bash
python src/global_performance.py --entity <wandb_entity> --project <wandb_project>
```

### 8) Weight Initialization Analysis (Zeros vs Xavier)

```bash
python src/weight_initialization.py --iterations 50 --activation_function sigmoid
```

### 9) Fixed Fashion-MNIST Runs (`best1`, `best2`, `best3`)

```bash
python src/fashion_mnist.py --epochs 10 --batch_size 128
```

## W&B Notes

- To run without syncing online:

```bash
--wandb_mode offline
```

- To disable W&B logging:

```bash
--wandb_mode disabled
```

Most scripts also accept:
- `--wandb_project`
- `--wandb_entity`
- `--wandb_run_name` (where applicable)

## Current Limitations

- `src/inference.py` is currently a stub and raises `NotImplementedError`.
