# Spezielle Aspekte der Wirtschaftsinformatik

This repository contains the code to our project in the course "Spezielle Aspekte der Wirtschaftsinformatik".

This project contains a multi-class image classifier for different kinds of animals using different CNN architectures.

## Authors
Marc Donauer

Philipp Kohnle

Tobias Nguyen

## Used dataset

This project uses the [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset from [Kaggle](https://www.kaggle.com/).

## Package structure
```
Project-ML-DL/
│── src/                            # Source code
│   │   ├── data_prep/              # Data preparation and augmentation
│   │   │   ├── __init__.py
│   │   │   ├── get_dataset.py
│   │   │   ├── split_dataset.py
│   │   │   ├── data_augmentation.py
│   │   ├── evaluation/             # Model evaluation
│   │   │   ├── __init__.py
│   │   │   ├── eval_custom.py
│   │   │   ├── eval_mobilenetv2.py
│   │   │   ├── eval_resnet50.py
│   │   │   ├── evaluation.py
│   │   │   ├── visualize_results.py
│   │   ├── models/                 # Models
│   │   │   ├── __init__.py
│   │   │   ├── build_custom.py
│   │   │   ├── build_mobilenetv2.py
│   │   │   ├── build_resnet50.py
│   │   │   ├── build.py
│   │   ├── utils/                  # Utility methods
│   │   │   ├── __init__.py
│   │   │   ├── cleanup.py
│   │   │   ├── config.py
│   │   │   ├── get_root_path.py
│   │   ├── main.py
│   │   ├── __init__.py
│── README.md                       # Documentation
│── requirements.txt                # Dependencies
│── .gitignore                      # Ignore files
│── LICENSE
```

## Platform Requirements

| Platform | GPU Support | Notes |
|----------|-------------|-------|
| **Linux** | ✅ CUDA | Native support, recommended |
| **Windows** | ✅ CUDA (via WSL) | Requires WSL 2.0 |
| **macOS** | ❌ CPU only | No CUDA support |

> **Windows Users:** Install [WSL 2.0](https://docs.microsoft.com/en-us/windows/wsl/install) and run the project inside WSL for GPU acceleration.

## Package installation with uv

[`uv`](https://docs.astral.sh/uv/) is an extremely fast Python package installer and resolver, written in Rust. It's recommended for its speed and reliability.

**Step 1: Install `uv`**

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

**Step 2: Create and activate virtual environment**

```bash
# Create a Python 3.12 virtual environment
uv venv --python 3.12

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

**Step 3: Install dependencies**

```bash
uv pip install -r requirements.txt
```

**Step 4: Install the package in editable mode**

```bash
uv pip install -e .
```

## Usage
```bash
# This command runs the entire pipeline from data gathering, augmentation, model building and evaluation
project
```