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
│── src/                   # Source code
│   │   ├── data_prep/
│   │   │   ├-- __init__.py
│   │   ├── evaluation/
│   │   │   ├-- __init__.py
│   │   ├── models/
│   │   │   ├-- __init__.py
│   │   ├── utils/
│   │   │   ├-- __init__.py
│   │   ├-- main.py
│   │   ├-- __init__.py
│── README.md              # Documentation
│── requirements.txt       # Dependencies
│── .gitignore             # Ignore files
│── LICENSE
```

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
# Install all dependencies from pyproject.toml
uv sync

# Or install with development dependencies
uv sync --all-extras

# Installation with pip if requirements.txt is used
uv pip install -r requirements.txt
```

**Step 4: Install the package in editable mode**

```bash
uv pip install -e .
```