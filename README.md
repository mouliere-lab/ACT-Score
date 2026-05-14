# ACT-score

## Overview

ACT score is a machine learning framework for early outcome prediction in large B-cell lymphoma (LBCL), including diffuse large B-cell lymphoma (DLBCL), using plasma cell-free DNA (cfDNA) collected during first-line treatment.

ACT stands for:

- **A**: Aberrations, represented by genomic tumor fraction
- **C**: Composition of cfDNA fragments, including fragment size and genome-wide fragmentation features
- **T**: Terminal motif analyses, including fragment-end motif diversity and motif-based scores

The pipeline is designed to integrate genomic and fragmentomic features from a single on-treatment plasma sample, typically collected after one cycle of therapy, to identify patients at higher risk of progression.

ACT integrates multiple cfDNA feature classes, including tumor fraction estimated by ichorCNA, fragment size distributions, motif-based features including the FrEIA score and Gini Diversity Index, LIONHEART-derived cfDNA tissue/cell-type contribution scores, and DELFI-FTK genome-wide fragmentation features. High-dimensional LIONHEART and DELFI-FTK features are selected using Elastic Net logistic regression within the training cohort only.

The model development framework uses a training cohort and an independent validation cohort. All feature selection, hyperparameter tuning, model fitting, and threshold optimization are performed using training data only. The validation cohort is used only for final evaluation.

## Repository structure

```text
ACT-score/
├── analysis/              # Command-line Python scripts
├── src/                   # Reusable pipeline functions
├── scripts/               # Bash scripts for running the workflow
├── data/                  # Example input files and feature lists
├── results/               # Output directory
├── environment.yml        # Conda environment file
├── INSTALL.md             # Installation instructions
└── README.md
```

## Input data format

The main input file should be a CSV file containing one row per sample and one column per feature.

By default, the pipeline expects the following metadata columns:

```text
subject_id
timepoint
cohort
2y_ttp
```

Default values used by the scripts are:

```text
timepoint = 1
file_name_col = subject_id
outcome_col = 2y_ttp
cohort_col = cohort
training_label = A
validation_label = B
```

The outcome column should contain:

```text
Yes
No
```

where:

```text
Yes = no progression by 2 years
No  = progression by 2 years
```

The cohort column should contain:

```text
A = training cohort
B = validation cohort
```

Example input columns:

```text
subject_id,timepoint,cohort,2y_ttp,Gini,ichorCNA,FrEIA_score,100–120,120–140,180–200,...
```

## Feature-list files

Feature names are provided using CSV files with one column called `feature`.

Example:

```csv
feature
Gini
ichorCNA
FrEIA_score
100–120
120–140
180–200
```

Recommended feature-list files:

```text
data/feature_lists/core_features.csv
data/feature_lists/LIONHEART_features.csv
data/feature_lists/DELFI-FTK_features.csv
```

The core feature file contains low-dimensional features used directly in the ACT model. LIONHEART and DELFI-FTK feature-list files contain the candidate high-dimensional features used for Elastic Net feature selection.

## System requirements

### Operating systems

The pipeline can be run on:

```text
macOS
Linux
Windows
```

The shell scripts are intended for macOS/Linux terminal use.

### Python version

Tested with:

```text
Python 3.12
```

### Hardware

The pipeline can be run on a standard computer for small example datasets. For real feature tables with many high-dimensional features, especially repeated model training and feature selection, use of multiple CPU cores is recommended.

The number of parallel jobs can be controlled using:

```text
--n_jobs
--nr_jobs
```

depending on the script.

## Installation

See `INSTALL.md` for full installation instructions.

Briefly:

```bash
conda env create -f environment.yml
conda activate ACT-ML
```

## Running the pipeline

The full workflow has three main steps:

1. Run feature selection for LIONHEART and DELFI-FTK features.
2. Train and validate ACT models using core features plus selected high-dimensional features.
3. Optionally apply trained models to new samples.

## Step 1: Feature selection

Feature selection is performed using Elastic Net logistic regression on the training cohort only.

Run:

```bash
bash scripts/run_feature_selection.sh
```

This script runs feature selection for:

```text
LIONHEART
DELFI-FTK
```

and saves output files to:

```text
results/feature_selection/
```

Expected output files include:

```text
LIONHEART_feature_selection_run_results.csv
LIONHEART_selected_feature_counts.csv
LIONHEART_selected_features_min1.csv
LIONHEART_all_selected_features_by_random_state.csv

DELFI-FTK_feature_selection_run_results.csv
DELFI-FTK_selected_feature_counts.csv
DELFI-FTK_selected_features_min5.csv
DELFI-FTK_all_selected_features_by_random_state.csv
```

The selected feature files are then used by the model training script.

You can also run feature selection manually:

```bash
python analysis/feature_selection.py \
  --feature_group LIONHEART \
  --input_file data/input_feature_table.csv \
  --feature_file data/feature_lists/LIONHEART_features.csv \
  --output_dir results \
  --min_times_selected 1 \
  --feature_col feature \
  --random_states 10 \
  --timepoint 1 \
  --outcome_col 2y_ttp \
  --cohort_col cohort \
  --training_label A \
  --validation_label B \
  --n_jobs 16
```

For DELFI-FTK:

```bash
python analysis/feature_selection.py \
  --feature_group DELFI-FTK \
  --input_file data/input_feature_table.csv \
  --feature_file data/feature_lists/DELFI-FTK_features.csv \
  --output_dir results \
  --min_times_selected 5 \
  --feature_col feature \
  --random_states 10 \
  --timepoint 1 \
  --outcome_col 2y_ttp \
  --cohort_col cohort \
  --training_label A \
  --validation_label B \
  --n_jobs 16
```

## Step 2: Train and validate ACT models

After feature selection, run:

```bash
bash scripts/run_train_validate.sh
```

This script reads:

```text
data/feature_lists/core_features.csv
results/feature_selection/LIONHEART_selected_features_min1.csv
results/feature_selection/DELFI-FTK_selected_features_min5.csv
```

and combines them into one ACT feature set.

The script trains and validates the following classifiers:

```text
Logistic_Regression
Random_Forest
Gaussian_Naive_Bayes
SVM
EasyEnsemble
Gradient_Boosting
```

Each classifier is trained across multiple random seeds. For each seed, the model is trained using the training cohort only, and validation performance is evaluated on the independent validation cohort. After all seeds are complete, predicted probabilities and performance metrics are averaged across runs.

Expected output structure:

```text
results/ACT_score_scale_True/
├── Logistic_Regression/
│   ├── rs_1/
│   ├── rs_2/
│   └── Average/
├── Random_Forest/
├── Gaussian_Naive_Bayes/
├── SVM/
├── EasyEnsemble/
└── Gradient_Boosting/
```

Each random-state folder contains:

```text
model .joblib file
prediction CSV
performance metrics CSV
feature importance CSV for Logistic Regression
```

The `Average/` folder contains averaged predictions and averaged performance metrics across random seeds.

You can also run one classifier manually:

```bash
python analysis/train_validate.py \
  --features Gini ichorCNA FrEIA_score "100–120" "120–140" "180–200" \
  --classifier_name Logistic_Regression \
  --timepoint 1 \
  --file_name_col subject_id \
  --outcome_col 2y_ttp \
  --cohort_col cohort \
  --training_label A \
  --validation_label B \
  --cv 5 \
  --cv_strategy StratifiedKFold \
  --scoring_method roc_auc \
  --threshold Optimal \
  --input_file_path data/input_feature_table.csv \
  --path_to_save_results results/ACT_score_scale_True \
  --rs 1 \
  --nr_jobs 16 \
  --scale_status True
```

Then average results across random seeds:

```bash
python analysis/average_runs.py \
  --classifier Logistic_Regression \
  --timepoint 1 \
  --outcome_col 2y_ttp \
  --cohort_col cohort \
  --validation_label B \
  --results-path results/ACT_score_scale_True \
  --cv 5 \
  --cv_strategy StratifiedKFold \
  --scoring_method roc_auc \
  --threshold Optimal \
  --rs 10 \
  --scale_status True
```

## Step 3: Predict new samples

To apply trained models to new samples:

```bash
bash scripts/run_predict_new_samples.sh
```

Or run manually:

```bash
python analysis/predict_new_samples.py \
  --classifier_name Logistic_Regression \
  --timepoint 1 \
  --file_name_col subject_id \
  --outcome_col 2y_ttp \
  --results_path results/ACT_score_scale_True \
  --input_file_path data/example_new_samples.csv \
  --output_predictions_path results/new_sample_predictions.csv \
  --output_performance_path results/new_sample_performance.csv \
  --cv 5 \
  --cv_strategy StratifiedKFold \
  --scoring_method roc_auc \
  --rs 10 \
  --scale_status True
```

If the new sample file does not contain the outcome column, the script will save predictions only. If the outcome column is present, it will also calculate performance metrics.

## Model training details

The pipeline uses repeated model development across random seeds. For each random seed:

1. Training samples are shuffled.
2. Hyperparameters are optimized using cross-validation within the training cohort.
3. The model is fitted using training data only.
4. The classification threshold is optimized using training data only when `threshold=Optimal`.
5. The trained model is applied to the validation cohort.
6. Predictions and performance metrics are saved.

Across random seeds, predicted probabilities are averaged, and the optimized thresholds are averaged to generate final binary predictions.

## Feature selection details

High-dimensional LIONHEART and DELFI-FTK features are selected separately using Elastic Net logistic regression. Feature selection is repeated across random seeds. Features with non-zero coefficients are counted across runs, and features selected at least a user-defined number of times are retained.

Default thresholds:

```text
LIONHEART: selected at least 1 time
DELFI-FTK: selected at least 5 times
```

These can be changed using:

```text
--min_times_selected
```

## Output files

Main output files include:

```text
*_predictions_*.csv
*_performance_metrics_*.csv
*_feature_importances_*.csv
*_average_predictions_*.csv
*_average_performance_metrics_*.csv
*_average_feature_importances_*.csv
```

For Logistic Regression, feature coefficients are saved. If feature scaling is enabled, features are ranked by absolute coefficient. If scaling is disabled, features are ranked by the absolute value of coefficient multiplied by feature standard deviation.

## Demo data

The repository may include small example CSV files to demonstrate expected input formatting. These files should not contain patient-identifiable or sensitive clinical data.

For real data analysis, replace:

```text
data/input_feature_table.csv
```

with your own private local file path.

## Important notes

- Do not rely on row order when combining clinical annotation and feature tables.
- Always merge or verify samples using a stable identifier such as `subject_id`.
- The input CSV should have the correct header row. If a table title row is present above the column names, remove it before running the pipeline.
- Column names are case-sensitive. For example, `2y_ttp` and `2Y_TTP` are different.
- Outcome values must be exactly `Yes` and `No`.
- Cohort labels must match the values passed through `--training_label` and `--validation_label`.
- For reproducible comparisons across machines, use the same input files, feature-list files, package versions, and random seeds. Numerical differences can still occur across systems because of different low-level math libraries or parallel execution.

## Citation

If you use this pipeline, please cite the associated manuscript.

## License

Please see the repository license file.
