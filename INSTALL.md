# Installation Guide

## Prerequisites

Before installing the ACT-score pipeline, install one of the following:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)

## Step-by-step instructions

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
conda env create -f ACT_score.yml
```

### 3. Activate the environment

```sh
conda activate ACT-ML
```
All commands should be run from the root of the `ACT-score` repository. If you cloned the repository as described above, you should already be in the correct directory.

### 4. Verify the installation

```sh
python --version
python -c "import pandas, sklearn, imblearn, joblib; print('Installation successful')"
```

## Install time

Installation typically takes about 5 to 10 minutes on a standard desktop computer, depending on internet speed and whether Python dependencies are already cached.

## Notes

The pipeline scripts assume that they are run from the root of the `ACT-score` repository and that the Conda environment has already been activated.

Before running any pipeline script, make sure you are in the project directory and the environment is active:

```sh
conda activate ACT-ML
cd /path/to/ACT-score
```

If you encounter the following error:

```sh
CondaError: Run 'conda init' before 'conda activate'
```

initialize Conda for your shell, for example:

```sh
conda init bash
```

Then close and reopen your terminal and activate the environment again:

```sh
conda activate ACT-ML
```

If you created the environment with a different name, activate that environment instead of `ACT-ML`.

