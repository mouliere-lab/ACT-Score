# Author: Parisa Mapar

import os
import sys
import warnings
import logging
import argparse
import numpy as np
import joblib

# Suppress warnings for cleaner command-line output
warnings.filterwarnings('ignore')

# Get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory of the script
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Ensure the parent directory is in the Python path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.classifier_info import get_classifier_info
from src.load_data import load_data
from src.data_processing import preprocess_data
from src.train import train_classifier
from src.performance_metrics import performance_metrics
from src.feature_importance import calculate_feature_importances, save_feature_importances
from src.evaluate import test_classifier


def train_validate(
    features,
    classifier_name,
    timepoint,
    file_name_col,
    outcome_col,
    cohort_col,
    training_label,
    validation_label,
    cv,
    cv_strategy,
    scoring_method,
    threshold,
    input_file_path,
    path_to_save_results,
    rs,
    nr_jobs,
    scale_status):
    """
    Train a classifier, generate predictions, and evaluate performance on the validation cohort.

    Parameters:
    -----------
        features : list of str
            List of feature names.
        classifier_name : str
            Name of the classifier to train.
        timepoint : int
            Timepoint to include in the analysis.
        file_name_col : str
            Name of the column containing sample/file identifiers.
        outcome_col : str
            Name of the outcome column. Expected values are "Yes" and "No".    
        cohort_col : str
            Name of the column containing training/validation cohort labels.
        training_label : str
            Label used for the training cohort.
        validation_label : str
            Label used for the validation cohort.    
        cv: int
           Number of cross-validation folds.  
        cv_strategy : str
            Cross-validation strategy used during model training.   
        scoring_method : str
            Scoring metric used for hyperparameter optimization.    
        threshold : str
            Classification threshold strategy. Use "Default" for 0.5 or "Optimal" for the optimized threshold.    
        input_file_path : str
            Path to the input feature table.
        path_to_save_results : str
            Directory where output files will be saved.
        rs : int
           Random seed used for shuffling the training set.
        nr_jobs : int
            Number of parallel jobs.
        scale_status : bool
            Whether to apply feature scaling during model training.    

    Returns:
    --------
        None
        Saves the fitted model, predictions, performance metrics, and logistic regression coefficients when applicable.
    """
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Classifier: {classifier_name}")
    logging.info(f"Random State: {rs}")
    
    # Get classifier parameters and create result directory
    classifier_info = get_classifier_info(classifier_name, nr_jobs)
    results_path_classifier = os.path.join(path_to_save_results, classifier_name,f'rs_{rs}')
    os.makedirs(results_path_classifier, exist_ok=True)

    try:
        
        # Load and preprocess data
        data = preprocess_data(
            load_data(input_file_path),
            timepoint=timepoint,
            file_name_col=file_name_col,
            outcome_col=outcome_col,
            cohort_col=cohort_col,
            training_label=training_label,
            validation_label=validation_label)
            
        data = data.assign(prediction_proba_1=np.nan, prediction=np.nan)
            
        # Prepare training data
        train_set = data[data[cohort_col] == training_label].sample(frac=1, random_state=rs).reset_index(drop=True)

        if train_set.empty:
            raise ValueError(
                f"No training samples found after filtering. "
                f"Check cohort_col={cohort_col}, training_label={training_label}, "
                f"timepoint={timepoint}, and outcome_col={outcome_col}.")
        
        target_train = np.array([1 if v == "No" else 0 for v in train_set[outcome_col]])

        # Train the classifier
        best_classifier, optimal_threshold = train_classifier(
            train_set=train_set,
            features=features,
            target_train=target_train,
            classifier_info=classifier_info,
            classifier_name=classifier_name,
            cv_strategy=cv_strategy,
            scoring_method=scoring_method,
            nr_jobs=nr_jobs,
            cv_nr=cv,
            scale=scale_status,
        )
        # Save fitted model and metadata
        model_bundle = {
            "model": best_classifier,
            "timepoint": timepoint,
            "features": features,
            "optimal_threshold": optimal_threshold,
            "classifier_name": classifier_name,
            "random_state": rs,
            "scale_status": scale_status
        }
        
        joblib.dump(
            model_bundle,
            os.path.join(
                results_path_classifier,
                f"{classifier_name}_model_cv_{cv}_{cv_strategy}_{scoring_method}_{scale_status}_scale_rs_{rs}.joblib"
            )
        )

            
        # Save logistic regression coefficients.
        if classifier_name == "Logistic_Regression":
            importances = calculate_feature_importances(
                cf=best_classifier,
                classifier_name=classifier_name,
            )
        
            save_feature_importances(
                importances=importances,
                features=features,
                train_set=train_set,
                results_path_classifier=results_path_classifier,
                classifier_name=classifier_name,
                title=f"cv_{cv}_rs_{rs}_{scale_status}_scale",
                scale_status=scale_status,
            )    
            

        # Generate predictions for all samples and save them
        y_test_pred, y_test_proba = test_classifier(best_classifier, data, features)
        data['prediction'], data['prediction_proba_1'] = y_test_pred, y_test_proba
        data.to_csv(os.path.join(results_path_classifier, f'{classifier_name}_predictions_cv_{cv}_{cv_strategy}_{scoring_method}_threshold_{threshold}_{scale_status}_scale_rs_{rs}.csv'), index=False)


        # Calculate and save performance metrics for the validation cohort
        val_data = data[data[cohort_col] == validation_label]

        if val_data.empty:
            raise ValueError(
                f"No validation samples found after filtering. "
                f"Check cohort_col={cohort_col}, validation_label={validation_label}, "
                f"timepoint={timepoint}, and outcome_col={outcome_col}.")

        if threshold == "Default":
            selected_threshold = 0.5
        elif threshold == "Optimal":
            selected_threshold = optimal_threshold
        else:
            raise ValueError("threshold must be either 'Default' or 'Optimal'")
        
        perf_metrics_val = performance_metrics(val_data, selected_threshold, outcome_col=outcome_col)

        perf_metrics_val.to_csv(os.path.join(results_path_classifier, f'{classifier_name}_performance_metrics_cv_{cv}_{cv_strategy}_{scoring_method}_threshold_{threshold}_{scale_status}_scale_rs_{rs}_tp{timepoint}.csv'), index=False)

    except Exception as e:
        logging.error(f"Error in classification: {e}")
        raise


def main():

    parser = argparse.ArgumentParser(description='Perform training and validation for a given classifier.')
    parser.add_argument('--features', nargs='+', type=str, required=True, help='List of feature names')
    parser.add_argument('--classifier_name', type=str, required=True, help='Name of the classifier')
    parser.add_argument('--timepoint', type=int, default=1, help="Timepoint to include in the analysis. Default: 1")
    parser.add_argument('--file_name_col', type=str, default='subject_id', help='Name of the sample/file identifier column')
    parser.add_argument('--outcome_col', type=str, default='2y_ttp', help='Name of the outcome column')
    parser.add_argument('--cohort_col', type=str, default='cohort', help='Name of the training/validation cohort column')
    parser.add_argument('--training_label', type=str, default='A', help='Label used for the training cohort')
    parser.add_argument('--validation_label', type=str, default='B', help='Label used for the validation cohort')
    parser.add_argument('--cv', type=int, default=4, help='Number of folds for cross-validation')
    parser.add_argument('--cv_strategy', type=str, required=True, help='Cross-validation strategy')
    parser.add_argument('--scoring_method', type=str, required=True, help='Scoring method for optimizing the classifier')
    parser.add_argument('--threshold', type=str, required=True, help='Classification threshold')
    parser.add_argument('--input_file_path', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--path_to_save_results', type=str, required=True, help='Path to save results')
    parser.add_argument('--rs', type=int, required=True, help='Random state')
    parser.add_argument('--nr_jobs', type=int, required=True, help='Number of parallel jobs')
    parser.add_argument('--scale_status', type=lambda x: x.lower() == 'true', required=True, help='Whether to scale or not')
        

    args = parser.parse_args()
    train_validate(
        args.features,
        args.classifier_name,
        args.timepoint,
        args.file_name_col,
        args.outcome_col,
        args.cohort_col,
        args.training_label,
        args.validation_label,
        args.cv,
        args.cv_strategy,
        args.scoring_method,
        args.threshold,
        args.input_file_path,
        args.path_to_save_results,
        args.rs,
        args.nr_jobs,
        args.scale_status,
    )
    
if __name__ == "__main__":
    main()

# Run this script from the terminal using:
# Example:
# python analysis/train_validate.py \
#   --features feature_1 feature_2 \
#   --classifier_name Logistic_Regression \
#   --timepoint 1 \
#   --file_name_col subject_id \
#   --outcome_col 2y_ttp \
#   --cohort_col cohort \
#   --training_label A \
#   --validation_label B \
#   --cv 5 \
#   --cv_strategy StratifiedKFold \
#   --scoring_method roc_auc \
#   --threshold Optimal \
#   --input_file_path data/example_input.csv \
#   --path_to_save_results results/ \
#   --rs 1 \
#   --nr_jobs 16 \
#   --scale_status True
    
    
    
    
    
    
    
    




