#!/bin/bash

set -euo pipefail

# Activate environment
source /Users/pmapar/miniconda3/etc/profile.d/conda.sh
conda activate ACT-ML

# Go to repository root
cd /path/to/ACT-score

# Settings
CLASSIFIER_NAME="Logistic_Regression"

TIMEPOINT=1
FILE_NAME_COL="subject_id"
OUTCOME_COL="2y_ttp"

CV=5
CV_STRATEGY="StratifiedKFold"
SCORING_METHOD="roc_auc"
RS_MAX=10
SCALE_STATUS=True

RESULTS_PATH="results/ACT_score_scale_${SCALE_STATUS}"
INPUT_FILE="data/example_new_samples.csv"

OUTPUT_PREDICTIONS="results/new_sample_predictions.csv"
OUTPUT_PERFORMANCE="results/new_sample_performance.csv"

python analysis/predict_new_samples.py \
  --classifier_name "$CLASSIFIER_NAME" \
  --timepoint "$TIMEPOINT" \
  --file_name_col "$FILE_NAME_COL" \
  --outcome_col "$OUTCOME_COL" \
  --results_path "$RESULTS_PATH" \
  --input_file_path "$INPUT_FILE" \
  --output_predictions_path "$OUTPUT_PREDICTIONS" \
  --output_performance_path "$OUTPUT_PERFORMANCE" \
  --cv "$CV" \
  --cv_strategy "$CV_STRATEGY" \
  --scoring_method "$SCORING_METHOD" \
  --rs "$RS_MAX" \
  --scale_status "$SCALE_STATUS"
