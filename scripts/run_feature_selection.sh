#!/bin/bash

set -euo pipefail

# Activate environment
source activate ACT-ML

# Go to repository root
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
