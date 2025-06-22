<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="license" />
  <img src="https://img.shields.io/badge/Built%20with-Python%203.12-blue.svg" alt="python" />
  <img src="https://img.shields.io/badge/Powered%20by-PyTorch%20%7C%20Diffusers%20%7C%20Accelerate-orange.svg" alt="powered by" />
</p>

<p align="center">
  <a href="docs/TFG_Utica_Maxim.pdf" download>
    <img src="https://img.shields.io/badge/📘%20Download%20TFG-TFG_Utica_Maxim.pdf-blue" alt="Download TFG"/>
  </a>
</p>


# Diffusion Models in Decentralized Federated Learning

This repository contains a modular simulation environment for training and evaluating **Denoising Diffusion Probabilistic Models (DDPMs)** in **Decentralized Federated Learning (DFL)** scenarios. The system was developed as part of a Bachelors's Thesis project focused on generative AI and distributed training.

## 🧠 Project Overview

**Title**: Implementation of Generative Diffusion Models in Decentralized Federated Learning  
**Author**: Maxim Utica Babyak  
**Degree**: Bachelor's in Computer Engineering  
**University**: Universidad de Murcia – Facultad de Informática  
**Date**: January 2025  
**Language**: Spanish  

This project explores how diffusion models can enhance the performance of decentralized federated learning systems, improving convergence, privacy, and robustness under non-IID data distributions.

You can read the full thesis here:  
📘 [TFG_Utica_Maxim.pdf](docs/TFG_Utica_Maxim.pdf)

## 📌 Key Features

* ✅ Simulation of decentralized topologies (e.g., ring, custom)
* 🧩 Modular codebase with YAML-configurable experiments
* ⟳ Decentralized training loop with model aggregation
* 🧠 Conditional DDPM generation with U-Net backbone
* 🔒 Label-sharing strategy for non-IID mitigation
* 📊 Built-in evaluation pipeline using auxiliary classifiers

## 📂 Project Structure

```
diffusion-dfl-implementation/
├── configs/
│   ├── ring/                     # Training & testing configs for ring topology
│   └── custom/                   # Training & testing configs for custom topology
│
├── src/
│   ├── core/
│   │   ├── cli.py                # CLI argument parsing
│   │   ├── config.py             # Config dataclasses and YAML loaders
│   │   ├── training.py           # Training logic
│   │   ├── testing.py            # Evaluation logic
│   │   ├── pipeline.py           # DDPM sampling pipeline
│   │   ├── launch.py             # Node orchestration and data loading
│   │   └── filesystem.py         # Utility functions for I/O, timers, etc.
│   └── data/
│       ├── classifier.py         # CNN classifier for evaluation
│       └── filtered_dataset.py   # Dataset wrapper with label/threshold filtering
│
├── laboratory/                   # Workspace for runtime data
│   ├── datasets/                 # Raw data downloaded via torchvision
│   ├── classifiers/              # Saved classifier weights
│   ├── topologies/               # neighbours.yaml and labels-*.yaml per topology
│   ├── scenarios/                # Training outputs (per run)
│   └── evaluations/              # Evaluation outputs (per model)
│
├── run.py                        # Script to run training or testing
├── pyproject.toml                # 📦 Dependency and build configuration
├── LICENSE                       # License (MIT)
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## 📦 Requirements

* Python 3.12
* **A CUDA 12.1-compatible NVIDIA GPU** (e.g., RTX 30xx or 40xx series)
* PyTorch with GPU support (cu121)

## 📥 Installation

This project uses [PEP 621](https://peps.python.org/pep-0621/) with `pyproject.toml`.

1. **Install PyTorch with CUDA 12.1 support**:

```bash
pip install torch torchvision torchmetrics -i https://download.pytorch.org/whl/cu121
```

2. **Install the project dependencies**:

```bash
pip install .
```

## ▶️ Running Experiments

All experiments are executed via the `entrypoint.py` module. Specify mode and configuration YAML:

### 🔹 Training

```bash
python run.py train mnist ring
python run.py test emnist ring --split letters
```

### 🔹 Evaluation

```bash
python -m src.entrypoint test --config configs/custom/test_emnist_letters.yaml
```

## 📈 Results

Empirical results demonstrate strong convergence and generalization under decentralized training:

| Dataset        | Accuracy (Best) |
| -------------- | --------------- |
| MNIST          | 98.6%           |
| FashionMNIST   | 91.79%          |
| EMNIST Letters | 90.49%          |

These results were achieved on 10-node topologies with label-sharing enabled.

## 📚 Academic Reference

If you use this project in academic work, please cite this repository. A BibTeX entry will be available upon request.

## 🧪 Future Work

* Secure aggregation protocols
* Asynchronous and hierarchical topologies
* Integration of differential privacy
* Real-time federated learning agents

## 🧾 License

This project is licensed under the MIT License.