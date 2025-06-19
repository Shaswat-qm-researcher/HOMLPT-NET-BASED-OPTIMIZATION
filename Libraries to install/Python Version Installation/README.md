# Python Environment Setup for Machine Learning Project

This folder contains an environment setup Python using Conda to ensure reproducibility and compatibility across systems. The environment includes TensorFlow, Scikit-learn, NumPy, Pandas, Matplotlib, and Joblib.

## Environment Details

- **Python version**: 3.12.9 (via Anaconda)
- **TensorFlow**: 2.19.0
- **Keras (via TensorFlow)**: 3.10.0
- **Scikit-learn**: 1.6.1
- **NumPy**: 2.1.3
- **Pandas**: 2.2.3
- **Matplotlib**: 3.10.1
- **Joblib**: 1.4.2

## Setup Instructions

### Step 1: Install [Anaconda](https://www.anaconda.com/products/distribution) if not already installed.

### Step 2: Create the Environment

Open a terminal (or Anaconda Prompt on Windows) and run:

```bash
conda env create -f environment.yml
```

### Step 3: Activate the Environment

```bash
conda activate myenv
```

## Verify the Installation

Run the following command to check Python and package versions:

```bash
python --version
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
```

## Files

- `environment.yml` — Conda environment definition file.
- `requirements.txt` — Optional: pip-based installation if using virtualenv instead of Conda.

## Note

Using Conda ensures exact Python version compatibility, especially useful when working with packages like TensorFlow or compiling on specific architectures.

---

Feel free to fork, clone, and use this setup for your own projects!
