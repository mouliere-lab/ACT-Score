# Installation Guide

## Prerequisites

Before installing the ACT-score pipeline, install one of the following:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)

Alternatively, the pipeline can also be installed using a standard Python virtual environment.

## Option 1: Installation with Conda

### 1. Clone the repository

```sh
git clone https://github.com/mouliere-lab/ACT-score.git
cd ACT-score
```

If you downloaded the repository as a ZIP file from GitHub, the extracted folder may be named `ACT-score-main`. In that case, enter that folder instead:

```sh
cd ACT-score-main
```

### 2. Create the Conda environment

```sh
conda create -n ACT-ML
```

### 3. Activate the environment

```sh
conda activate ACT-ML
```

### 4. Install dependencies

```sh
pip install -r requirements.txt
```

### 5. Verify the installation

```sh
python --version
python -c "import pandas, sklearn, imblearn, joblib; print('Installation successful')"
```

## Option 2: Installation with Python venv

If you do not use Conda, you can install the pipeline using `venv`.

### 1. Clone the repository

```sh
git clone https://github.com/mouliere-lab/ACT-score.git
cd ACT-score
```

### 2. Create and activate a virtual environment

```sh
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, use:

```sh
.venv\Scripts\activate
```

### 3. Install dependencies

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the installation

```sh
python --version
python -c "import pandas, sklearn, imblearn, joblib; print('Installation successful')"
```

## Install time

Installation typically takes about 5 to 10 minutes on a standard desktop computer, depending on internet speed and whether Python dependencies are already cached.

## Notes

The bash scripts in the `scripts/` directory assume that the environment is called `ACT-ML`. If you use a different environment name, update the activation line in the scripts accordingly.

For example, with Conda:

```sh
conda activate ACT-ML
```

or with `venv`:

```sh
source .venv/bin/activate
```
