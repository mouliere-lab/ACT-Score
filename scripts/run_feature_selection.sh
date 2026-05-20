#!/bin/bash

set -euo pipefail

echo "============================================================"
echo "Starting ACT feature selection"
echo "============================================================"

echo "Using active Conda environment: ${CONDA_DEFAULT_ENV:-none}"

if [ "${CONDA_DEFAULT_ENV:-}" != "ACT-ML" ]; then
    echo "Error: expected Conda environment 'ACT-ML', but current environment is '${CONDA_DEFAULT_ENV:-none}'."
    echo "Please activate the environment before running this script:"
    echo "conda activate ACT-ML"
    exit 1
fi

# Go to repository root
# Replace this with the path to your cloned ACT-score repository.
cd /path/to/ACT-score

# User settings
INPUT_FILE="data/input_feature_table.csv"
FEATURE_LIST_DIR="data/feature_lists"
OUTPUT_DIR="results"

TIMEPOINT=1
OUTCOME_COL="2y_ttp"
COHORT_COL="cohort"
TRAINING_LABEL="A"
VALIDATION_LABEL="B"
FEATURE_COL="feature"
N_JOBS=16
RANDOM_STATES=10

echo "Checking input files"
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: input file not found: $INPUT_FILE"
    exit 1
fi

if [ ! -f "$FEATURE_LIST_DIR/LIONHEART_features.csv" ]; then
    echo "Error: LIONHEART feature file not found: $FEATURE_LIST_DIR/LIONHEART_features.csv"
    exit 1
fi

if [ ! -f "$FEATURE_LIST_DIR/DELFI-FTK_features.csv" ]; then
    echo "Error: DELFI-FTK feature file not found: $FEATURE_LIST_DIR/DELFI-FTK_features.csv"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Running LIONHEART feature selection"
echo "============================================================"


# LIONHEART feature selection
python analysis/feature_selection.py \
  --feature_group LIONHEART \
  --input_file "$INPUT_FILE" \
  --feature_file "$FEATURE_LIST_DIR/LIONHEART_features.csv" \
  --output_dir "$OUTPUT_DIR" \
  --min_times_selected 1 \
  --feature_col "$FEATURE_COL" \
  --random_states "$RANDOM_STATES" \
  --timepoint "$TIMEPOINT" \
  --outcome_col "$OUTCOME_COL" \
  --cohort_col "$COHORT_COL" \
  --training_label "$TRAINING_LABEL" \
  --validation_label "$VALIDATION_LABEL" \
  --n_jobs "$N_JOBS"


echo "Finished LIONHEART feature selection"

echo "============================================================"
echo "Running DELFI-FTK feature selection"
echo "============================================================"


# DELFI-FTK feature selection
python analysis/feature_selection.py \
  --feature_group DELFI-FTK \
  --input_file "$INPUT_FILE" \
  --feature_file "$FEATURE_LIST_DIR/DELFI-FTK_features.csv" \
  --output_dir "$OUTPUT_DIR" \
  --min_times_selected 5 \
  --feature_col "$FEATURE_COL" \
  --random_states "$RANDOM_STATES" \
  --timepoint "$TIMEPOINT" \
  --outcome_col "$OUTCOME_COL" \
  --cohort_col "$COHORT_COL" \
  --training_label "$TRAINING_LABEL" \
  --validation_label "$VALIDATION_LABEL" \
  --n_jobs "$N_JOBS"

echo "Finished DELFI-FTK feature selection"

echo "============================================================"
echo "Feature selection completed successfully"
echo "Outputs saved in: $OUTPUT_DIR/feature_selection"
echo "============================================================"
  
