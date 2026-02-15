# StaR: Stateful Dynamic-Graph Root Cause Analysis through Memory-Enhanced Causality Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Anonymous submission for peer review**

## Overview

**StaR (Stateful Root Cause Analysis)** is a novel deep learning framework for root cause analysis (RCA) and causal discovery in dynamic multivariate time series systems. By integrating temporal graph networks with Granger causality, StaR effectively captures both temporal dependencies and graph-structured relationships for accurate fault localization and causal structure inference.

### Key Features

- **Temporal Memory Mechanism**: Tracks evolving node states over time using GRU/LSTM/Transformer/MLP-based memory updaters
- **Message Passing**: Propagates information between nodes to capture inter-variable dependencies
- **Granger Causality Discovery**: Neural-network-based Granger causality for learning causal relationships
- **Dynamic Graph Support**: Handles time-varying graph structures common in real-world systems
- **Comprehensive Evaluation**: Includes both root cause analysis (AC@k) and causal discovery (F1, AUROC, AUPRC) metrics

## Architecture

StaR consists of three main components:

1. **StaRGC (Stateful Graph with Granger Causality)**: Core temporal graph module with:
   - Temporal memory for node state tracking
   - Message passing for information propagation
   - Granger causality layer for causal relationship learning

2. **StaR Encoder-Decoder**: AERCA-style architecture with:
   - Encoder: Transforms observations to latent exogenous variables
   - Decoder: Reconstructs observations from latent variables
   - Sparsity and smoothness regularization

3. **Model Variants**:
   - **Baseline AERCA**: Original model without temporal mechanisms
   - **StaR**: Full model with temporal memory and message passing
   - **Ablation versions**: For studying individual component contributions

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.9.0+
- CUDA 13.0+ (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd TGN_AERCA_public
```

2. **Create virtual environment** (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or with `uv` (faster):
```bash
pip install uv
uv pip install -r requirements.txt
```

## Quick Start

### 1. Train StaR Model

```bash
# Train on linear dataset
python main_star.py --dataset_name linear

# Train on Lorenz96 dataset
python main_star.py --dataset_name lorenz96

# Train on random connection dataset with dynamic graphs
python main_star.py --dataset_name random_connection --base_connection_prob 0.65 --preprocessing_data 1
```

### 2. Train Baseline AERCA

```bash
# Train original AERCA (without temporal mechanisms)
python main.py --dataset_name linear
```

### 3. Run Ablation Studies

```bash
# Disable message passing (test temporal memory alone)
python main_star.py --dataset_name linear --disable_message_passing

# Test different memory mechanisms
python main_star.py --dataset_name linear --memory_updater_type lstm
python main_star.py --dataset_name linear --memory_updater_type transformer
python main_star.py --dataset_name linear --memory_updater_type mlp

# Test different memory dimensions
python main_star.py --dataset_name linear --memory_dim 32
python main_star.py --dataset_name linear --memory_dim 128
```

## Datasets

The repository includes data loading and generation code for the following datasets:

### Synthetic Datasets
- **Linear**: Linear Granger causal relationships
- **Nonlinear**: Nonlinear causal relationships  
- **Lorenz96**: Chaotic dynamical system
- **Lotka-Volterra**: Predator-prey dynamics
- **Random Connection**: Dynamic graph with time-varying connections
- **Temporal Service**: Microservice dependency graphs

### Real-World Datasets
- **SWaT** (Secure Water Treatment): Industrial control system
- **MSDS** (Mars Science Data System): Spacecraft telemetry
- **TEP** (Tennessee Eastman Process): Chemical process control
- **MSL** (Mars Science Laboratory): NASA spacecraft data
- **SMD** (Server Machine Dataset): Server monitoring metrics
- **AIOps**: IT operations data

**Note**: For real-world datasets, you'll need to download the data separately and place it in the `datasets/<dataset_name>/` directory. See individual dataset classes in `datasets/` for specific instructions.

## Running Experiments

### Sequential Experiments

Run multiple datasets sequentially:

```bash
# PowerShell
.\sensitivity_ablation_scripts\run_experiments_sequential.ps1

# Bash
bash sensitivity_ablation_scripts/run_experiments_sequential.sh
```

### Comprehensive Ablation Study

Run all ablation studies (memory dimension, memory mechanism, message passing, etc.):

```bash
.\sensitivity_ablation_scripts\run_all_ablation_studies.ps1
```

This will run:
1. Memory dimension ablation (32, 64, 128)
2. Memory mechanism ablation (GRU, LSTM, Transformer, MLP)
3. Message passing ablation (enabled vs disabled)
4. Time dimension ablation
5. Message dimension ablation

## Project Structure

```
.
├── models/
│   ├── aerca.py                    # Baseline AERCA model
│   ├── star.py                     # StaR model (main)
│   ├── star_gc.py                  # StaR-GC with GRU memory
│   ├── star_gc_flexible.py         # StaR-GC with flexible memory updaters
│   ├── memory_updaters.py          # GRU/LSTM/Transformer/MLP updaters
│   └── senn.py                     # SENN-GC module
├── utils/
│   ├── utils.py                    # Evaluation metrics and utilities
│   ├── model_manager.py            # Model saving/loading
│   ├── dynamic_graph_metrics.py    # Dynamic graph evaluation
│   └── tds_metric.py               # Temporal Decoupling Score
├── datasets/
│   ├── linear.py                   # Linear dataset
│   ├── nonlinear.py                # Nonlinear dataset
│   ├── lorenz96.py                 # Lorenz96 dataset
│   ├── random_connection_service.py # Dynamic graph dataset
│   └── ...                         # Other datasets
├── args/
│   └── *_args.py                   # Dataset-specific argument parsers
├── sensitivity_ablation_scripts/
│   ├── run_all_ablation_studies.ps1
│   ├── run_experiments_sequential.ps1
│   └── ...                         # Other experiment scripts
├── main_star.py                    # Train/evaluate StaR
├── main.py                         # Train/evaluate baseline AERCA
├── requirements.txt
└── README.md
```

## Configuration

### Common Arguments

**Dataset Selection**:
```bash
--dataset_name {linear,nonlinear,lorenz96,swat,msds,...}
--preprocessing_data {0,1}          # 1: generate new data, 0: load existing
```

**Model Architecture**:
```bash
--hidden_layer_size 128             # Hidden layer size
--num_hidden_layers 2               # Number of hidden layers
--window_size 5                     # Sliding window size (max lag)
```

**StaR-Specific Parameters**:
```bash
--memory_dim 64                     # Memory vector dimension
--time_dim 32                       # Temporal encoding dimension
--message_dim 64                    # Message dimension
--memory_updater_type {gru,lstm,transformer,mlp}
--disable_message_passing           # Ablation: disable message passing
```

**Training**:
```bash
--epochs 100                        # Number of training epochs
--lr 0.0001                         # Learning rate
--seed 42                           # Random seed
```

**Regularization**:
```bash
--encoder_lambda 0.5                # Encoder sparsity weight
--decoder_lambda 0.5                # Decoder sparsity weight
--encoder_gamma 0.5                 # Encoder smoothness weight
--decoder_gamma 0.5                 # Decoder smoothness weight
--beta 0.5                          # KL divergence weight
```

## Evaluation Metrics

### Root Cause Analysis
- **AC@k**: Accuracy at top-k (k=1,3,5,10)
- **AC*@k**: Alternative AC@k with different normalization
- **Avg@k**: Average across all k values
- **GAAC**: Graph-Aware Anomaly Contextualization (for dynamic graphs)
- **TDS**: Temporal Decoupling Score (for dynamic graphs)

### Causal Discovery
- **F1 Score**: Harmonic mean of precision and recall
- **AUROC**: Area under ROC curve
- **AUPRC**: Area under precision-recall curve
- **Hamming Distance**: Number of edge differences
- **SHD, FDR, TPR** (for dynamic graphs): Dynamic causal discovery metrics

## Results

Models are automatically saved in `saved_models/<dataset>/<method>/<timestamp>/` with:
- `model_state.pt`: Model weights
- `training_params.json`: Hyperparameters
- `thresholds.json`: Learned thresholds

Experiment logs are saved in `experiment_logs/` and `logs/`.

## Citation

```bibtex
@article{anonymous2026star,
  title={StaR: Stateful Dynamic-Graph Root Cause Analysis through Memory-Enhanced Causality Discovery},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on PyTorch and PyTorch Geometric
- Inspired by Temporal Graph Networks (TGN) and AERCA frameworks
- Uses various public datasets for evaluation (SWaT, MSDS, TEP, MSL, SMD, AIOps)

## Contact

For questions or issues, please open a GitHub issue or contact the authors (contact information will be added after review).
