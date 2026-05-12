import os
import sys
import logging
import numpy as np
import pandas as pd
import argparse
import joblib
import warnings

warnings.filterwarnings("ignore")

# Get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory of the script
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Ensure the parent directory is in the Python path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.load_data import load_data
from src.performance_metrics import performance_metrics
from src.evaluate import test_classifier

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def predict_new_samples(
    classifier_name,
    timepoint,
    file_name_col,
    outcome_col,
    results_path,
    input_file_path,
    output_predictions_path,
    output_performance_path,
    cv,
    cv_strategy,
    scoring_method,
    random_states,
    scale_status,
):
    """
    Apply saved models from multiple random states to new samples.

    The script loads one saved model per random state, predicts the probability
    of the positive class for each sample, averages probabilities across models,
    averages the optimized thresholds, and assigns a final binary prediction.

    Parameters
    ----------
    classifier_name : str
        Name of the classifier.
    timepoint : int, optional
        If provided, only samples from this timepoint are included.   
    file_name_col : str, default="file_name"
        Name of the sample/file identifier column.     
    outcome_col : str, default="2y_ttp"
        Name of the outcome column. Expected values are "Yes" and "No".      
    results_path : str
        Path to the directory containing saved model results.
    input_file_path : str
        Path to the input CSV file containing pre-computed features.
    output_predictions_path : str
        Path where averaged predictions will be saved.
    output_performance_path : str
        Path where performance metrics will be saved if outcome labels are available.
    cv : int
        Number of cross-validation folds used during model training.
    cv_strategy : str
        Cross-validation strategy used during model training.
    scoring_method : str
        Scoring metric used during model training.
    random_states : int
        Number of random-state models to average.
    scale_status : bool
        Whether feature scaling was used during model training.

    Returns
    -------
    None
        Saves averaged predictions and, if possible, performance metrics.
    """

    logging.info(f"Classifier: {classifier_name}")
    
    data = load_data(input_file_path)
    
    if outcome_col in data.columns:
        data = data[data[outcome_col].isin(["Yes", "No"])]

    if timepoint is not None:
        data = data[data["timepoint"] == timepoint]

    data = data.sort_values(by=[file_name_col]).reset_index(drop=True)

    results_path_classifier = os.path.join(results_path, classifier_name)

    merged_data = data.copy()
    optimal_thresholds = []
    features_reference = None

    # Loop through saved models
    for i in range(1, random_states + 1):
        model_file = os.path.join(
            results_path_classifier,
            f'rs_{i}',
            f"{classifier_name}_model_cv_{cv}_{cv_strategy}_{scoring_method}_{scale_status}_scale_rs_{i}.joblib"
        )

        try:
            bundle = joblib.load(model_file)
        except FileNotFoundError:
            logging.error(f"Model file not found: {model_file}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Unexpected error loading model file {model_file}: {e}")
            sys.exit(1)

        model = bundle["model"]
        features = bundle["features"]
        opt_threshold = bundle["optimal_threshold"]

        if features_reference is None:
            features_reference = features
        elif features != features_reference:
            logging.error(f"Feature mismatch in model file: {model_file}")
            sys.exit(1)

        _, prediction_proba = test_classifier(
            classifier=model,
            test_set=data.copy(),
            features=features,
        )
        
        merged_data[f'prediction_proba_1_rs_{i}'] = prediction_proba
        optimal_thresholds.append(float(opt_threshold))



    if merged_data is None or merged_data.empty:
        logging.error(f"No valid data found for classifier: {classifier_name}")
        return

    proba_cols = [f'prediction_proba_1_rs_{i}' for i in range(1, random_states + 1)]

    merged_data['average_prediction_proba_1'] = merged_data[proba_cols].mean(axis=1)

    mean_threshold = np.mean(optimal_thresholds)
    
    merged_data['prediction'] = np.where(
        merged_data['average_prediction_proba_1'] >= mean_threshold, 1, 0
    )

    os.makedirs(os.path.dirname(output_predictions_path), exist_ok=True)
    merged_data.to_csv(output_predictions_path, index=False)
    logging.info(f"Saved averaged predictions to {output_predictions_path}")


    if outcome_col in merged_data.columns and output_performance_path is not None:
        performance_data = merged_data.copy()
        performance_data["prediction_proba_1"] = performance_data["average_prediction_proba_1"]

        performance = performance_metrics(
            data_set=performance_data,
            optimal_threshold=mean_threshold,
            outcome_col=outcome_col,
        )

        os.makedirs(os.path.dirname(output_performance_path), exist_ok=True)
        performance.to_csv(output_performance_path, index=False)
        logging.info(f"Saved performance metrics to {output_performance_path}")




def main():
    parser = argparse.ArgumentParser(
        description="Apply saved models from multiple random states to new samples."
    )

    parser.add_argument("--classifier_name", type=str, required=True, help="Name of the classifier")
    parser.add_argument("--timepoint", type=int, default=1, help="Timepoint to filter")
    parser.add_argument("--file_name_col", type=str, default="subject_id", help="Name of the sample/file identifier column")
    parser.add_argument("--outcome_col", type=str, default="2y_ttp", help="Name of the outcome column")
    parser.add_argument("--results_path", type=str, required=True, help="Path to saved model results")
    parser.add_argument("--input_file_path", type=str, required=True, help="Path to input feature CSV")
    parser.add_argument("--output_predictions_path", type=str, required=True, help="Path to save averaged predictions")
    parser.add_argument("--output_performance_path", type=str, default=None, help="Path to save performance metrics")
    parser.add_argument("--cv", type=int, required=True, help="Number of CV folds")
    parser.add_argument("--cv_strategy", type=str, required=True, help="CV strategy used during training")
    parser.add_argument("--scoring_method", type=str, required=True, help="Scoring method used during training")
    parser.add_argument("--rs", type=int, required=True, help="Number of random-state models")
    parser.add_argument("--scale_status", type=lambda x: x.lower() == "true", required=True, help="Whether scaling was used")

    
    

    args = parser.parse_args()

    predict_new_samples(
        classifier_name=args.classifier_name,
        results_path=args.results_path,
        input_file_path=args.input_file_path,
        output_predictions_path=args.output_predictions_path,
        output_performance_path=args.output_performance_path,
        cv=args.cv,
        cv_strategy=args.cv_strategy,
        scoring_method=args.scoring_method,
        random_states=args.rs,
        scale_status=args.scale_status,
        timepoint=args.timepoint,
        outcome_col=args.outcome_col,
        file_name_col=args.file_name_col,
    )


if __name__ == "__main__":
    main()


# Example:
# python analysis/predict_new_samples.py \
#   --classifier_name Logistic_Regression \
#   --timepoint 1 \
#   --file_name_col subject_id \
#   --outcome_col 2y_ttp \
#   --results_path results/Final_results_scale_False \
#   --input_file_path data/example_new_samples.csv \
#   --output_predictions_path results/new_sample_predictions.csv \
#   --output_performance_path results/new_sample_performance.csv \
#   --cv 5 \
#   --cv_strategy StratifiedKFold \
#   --scoring_method roc_auc \
#   --rs 10 \
#   --scale_status False
