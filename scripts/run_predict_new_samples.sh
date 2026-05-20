#!/bin/bash

set -euo pipefail

echo "Using active Conda environment: ${CONDA_DEFAULT_ENV:-none}"

if [ "${CONDA_DEFAULT_ENV:-}" != "ACT-ML" ]; then
    echo "Error: expected Conda environment 'ACT-ML', but current environment is '${CONDA_DEFAULT_ENV:-none}'."
    echo "Please activate the environment before running this script:"
    echo "conda activate ACT-ML"
    exit 1
fi

# Settings
CLASSIFIER_NAME="Logistic_Regression"

TIMEPOINT=1
FILE_NAME_COL="subject_id"
OUTCOME_COL="2y_ttp"

CV=5
CV_STRATEGY="StratifiedKFold"
SCORING_METHOD="roc_auc"
RS_MAX=10
SCALE_STATUS=False

RESULTS_PATH="results/ACT_score"
INPUT_FILE="data/example_new_samples.csv"

if [ ! -d "$RESULTS_PATH" ]; then
    echo "Error: trained model/results directory not found: $RESULTS_PATH"
    echo "Please run scripts/run_train_validate.sh first."
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: input file not found: $INPUT_FILE"
    exit 1
fi

OUTPUT_DIR="results/new_samples"
mkdir -p "$OUTPUT_DIR"

OUTPUT_PREDICTIONS="$OUTPUT_DIR/new_sample_predictions.csv"
OUTPUT_PERFORMANCE="$OUTPUT_DIR/new_sample_performance.csv"

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

echo "Prediction completed successfully."
echo "Predictions saved to: $OUTPUT_PREDICTIONS"
if [ -f "$OUTPUT_PERFORMANCE" ]; then
    echo "Performance metrics saved to: $OUTPUT_PERFORMANCE"
else
    echo "No performance metrics file was created. This is expected if the input file does not contain the outcome column '$OUTCOME_COL'."
fi
