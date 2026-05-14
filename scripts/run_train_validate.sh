#!/bin/bash

set -euo pipefail

# Activate environment
conda activate ACT-ML

# Go to repository root
cd /path/to/ACT-score

# Classifiers
classifiers=(
  "Logistic_Regression"
  "Random_Forest"
  "Gaussian_Naive_Bayes"
  "SVM"
  "EasyEnsemble"
  "Gradient_Boosting"
)

# Feature-list CSV files.
# These files should contain a column called "feature".
core_feature_file="data/feature_lists/core_features.csv"
lionheart_feature_file="results/feature_selection/LIONHEART_selected_features_min1.csv"
delfi_feature_file="results/feature_selection/DELFI-FTK_selected_features_min5.csv"

# Read features from the CSV files

core_features=()
while IFS= read -r feature; do
    core_features+=("$feature")
done < <(
python - <<EOF
import pandas as pd
df = pd.read_csv("$core_feature_file")
for feature in df["feature"].dropna().astype(str):
    print(feature)
EOF
)

lionheart_features=()
while IFS= read -r feature; do
    lionheart_features+=("$feature")
done < <(
python - <<EOF
import pandas as pd
df = pd.read_csv("$lionheart_feature_file")
for feature in df["feature"].dropna().astype(str):
    print(feature)
EOF
)

delfi_features=()
while IFS= read -r feature; do
    delfi_features+=("$feature")
done < <(
python - <<EOF
import pandas as pd
df = pd.read_csv("$delfi_feature_file")
for feature in df["feature"].dropna().astype(str):
    print(feature)
EOF
)

# Combine all ACT features.
features=(
    "${core_features[@]}"
    "${lionheart_features[@]}"
    "${delfi_features[@]}"
)

if [ "${#features[@]}" -eq 0 ]; then
    echo "Error: no features were loaded. Check feature-list CSV files."
    exit 1
fi

# Inputs
INPUT_FILE="data/input_feature_table.csv"

# Run settings
TIMEPOINT=1
FILE_NAME_COL="subject_id"
OUTCOME_COL="2y_ttp"
COHORT_COL="cohort"
TRAINING_LABEL="A"
VALIDATION_LABEL="B"

CV=5
CV_STRATEGY="StratifiedKFold"
SCORING_METHOD="roc_auc"
THRESHOLD="Optimal"
RS_MAX=10
NR_JOBS=16
SCALE_STATUS=True

RUN_NAME="ACT_score"
RESULTS_DIR="results/${RUN_NAME}_scale_${SCALE_STATUS}"

mkdir -p "$RESULTS_DIR"
mkdir -p logs

echo "Number of features: ${#features[@]}"
echo "Results directory: $RESULTS_DIR"

for classifier_name in "${classifiers[@]}"; do

    echo "============================================================"
    echo "Running classifier: $classifier_name"
    echo "============================================================"

    for ((rs=1; rs<=RS_MAX; rs++)); do
        echo "Running classifier: $classifier_name | random seed: $rs"

        python analysis/train_validate.py \
            --features "${features[@]}" \
            --classifier_name "$classifier_name" \
            --timepoint "$TIMEPOINT" \
            --file_name_col "$FILE_NAME_COL" \
            --outcome_col "$OUTCOME_COL" \
            --cohort_col "$COHORT_COL" \
            --training_label "$TRAINING_LABEL" \
            --validation_label "$VALIDATION_LABEL" \
            --cv "$CV" \
            --cv_strategy "$CV_STRATEGY" \
            --scoring_method "$SCORING_METHOD" \
            --threshold "$THRESHOLD" \
            --input_file_path "$INPUT_FILE" \
            --path_to_save_results "$RESULTS_DIR" \
            --rs "$rs" \
            --nr_jobs "$NR_JOBS" \
            --scale_status "$SCALE_STATUS"
    done

    echo "Averaging results for classifier: $classifier_name"

    python analysis/average_runs.py \
        --classifier "$classifier_name" \
        --timepoint "$TIMEPOINT" \
        --outcome_col "$OUTCOME_COL" \
        --cohort_col "$COHORT_COL" \
        --validation_label "$VALIDATION_LABEL" \
        --results-path "$RESULTS_DIR" \
        --cv "$CV" \
        --cv_strategy "$CV_STRATEGY" \
        --scoring_method "$SCORING_METHOD" \
        --threshold "$THRESHOLD" \
        --rs "$RS_MAX" \
        --scale_status "$SCALE_STATUS"

done

echo "All classifiers completed."
