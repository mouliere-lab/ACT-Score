# Author: Parisa Mapar

import os
import sys
import logging
import numpy as np
import pandas as pd
import argparse

# Get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory of the script
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Ensure the parent directory is in the Python path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from src.performance_metrics import performance_metrics

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def average_results(
    classifier_name,
    timepoint,
    outcome_col,
    cohort_col,
    validation_label,
    results_path,
    cv,
    cv_strategy,
    scoring_method,
    threshold,
    random_states,
    scale_status,
):
    """
    Calculate the average of performance metrics over different runs with different random states.
    
    Parameters:
    -----------
        classifier_name (str): Name of the classifier to analyze.
        timepoint (int): Timepoint used in the analysis.
        outcome_col (str): Name of the outcome column.
        results_path (str): Path to the directory containing the results data for different random states.
        cv (int): number of cross-validation folds
        random_states (int): Number of random state iterations to consider for averaging.
        scale_status : Whether the data has been scaled or not
    
    Returns:
    --------
        None
        Saves an aggregated CSV file with average predictions and a CSV file with calculated performance metrics.
    """
    logging.info(f"Classifier: {classifier_name}")


    # Define paths
    results_path_classifier = os.path.join(results_path, classifier_name)
    results_path_classifier_avg = os.path.join(results_path_classifier, 'Average')
    os.makedirs(results_path_classifier_avg, exist_ok=True)

    merged_data = None
    optimal_thresholds = []
    feature_importance_dfs = []

    # Read and aggregate results over different random states
    for i in range(1, random_states + 1):
        pred_file = os.path.join(results_path_classifier, f'rs_{i}', f'{classifier_name}_predictions_cv_{cv}_{cv_strategy}_{scoring_method}_threshold_{threshold}_{scale_status}_scale_rs_{i}.csv')

        try:
            full_data = pd.read_csv(pred_file)
        except FileNotFoundError:
            logging.error(f"Predictions file not found: {pred_file}")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            logging.warning(f"Predictions file is empty: {pred_file}")
            sys.exit(1)
        except pd.errors.ParserError:
            logging.error(f"Error parsing Predictions file: {pred_file}.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Unexpected error with Predictions file {pred_file}: {e}")
            sys.exit(1)
            
        # Rename probability column for this random state
        full_data = full_data.copy()
        full_data = full_data.rename(
            columns={'prediction_proba_1': f'prediction_proba_1_rs_{i}'}
        )
        
        
        # Keep one shared set of metadata columns + this rs-specific probability column
        cols_to_keep = [col for col in full_data.columns if col != 'prediction']
        current_data = full_data[cols_to_keep]
        
        if merged_data is None:
            merged_data = current_data
        else:
            # Merge only the new probability column from this random state
            rs_col = f'prediction_proba_1_rs_{i}'
            merge_keys = [col for col in current_data.columns if col != rs_col]
            merge_keys = [col for col in merge_keys if col in merged_data.columns]

            merged_data = pd.merge(
                merged_data,
                current_data[merge_keys + [rs_col]],
                on=merge_keys,
                how='inner'
            )
    
            

        performance_metrics_path = os.path.join(
            results_path_classifier,
            f"rs_{i}",
            f"{classifier_name}_performance_metrics_cv_{cv}_{cv_strategy}_{scoring_method}_threshold_{threshold}_{scale_status}_scale_rs_{i}_tp{timepoint}.csv"
        )
        
        
        try:
            metrics = pd.read_csv(performance_metrics_path)
            opt_threshold=metrics['Optimal threshold'].iloc[0]
            optimal_thresholds.append(opt_threshold)
        except FileNotFoundError:
            logging.error(f"Performance_metrics file not found: {performance_metrics_path}")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            logging.warning(f"Performance_metrics file is empty: {performance_metrics_path}")
            sys.exit(1)
        except pd.errors.ParserError:
            logging.error(f"Error parsing Performance_metrics file: {performance_metrics_path}.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Unexpected error with Performance_metrics file {performance_metrics_path}: {e}")
            sys.exit(1)
            

        if classifier_name == "Logistic_Regression":            
            # Read feature importance file
            feature_importance_path = os.path.join(
                results_path_classifier,
                f'rs_{i}',
                f'{classifier_name}_feature_importances_cv_{cv}_rs_{i}_{scale_status}_scale.csv'
            )
    
            try:
                imp_df = pd.read_csv(feature_importance_path)
                feature_importance_dfs.append(imp_df.copy())
            except FileNotFoundError:
                logging.warning(f"Feature importance file not found: {feature_importance_path}")
            except pd.errors.EmptyDataError:
                logging.warning(f"Feature importance file is empty: {feature_importance_path}")
            except pd.errors.ParserError:
                logging.warning(f"Error parsing Feature importance file: {feature_importance_path}")
            except Exception as e:
                logging.warning(f"Unexpected error with Feature importance file {feature_importance_path}: {e}")
            


    if merged_data is None or merged_data.empty:
        logging.error(f"No valid data found for classifier: {classifier_name}")
        return

    # Identify all probability columns
    proba_cols = [f'prediction_proba_1_rs_{i}' for i in range(1, random_states + 1)]

    # Compute average probability across all random states
    merged_data['average_prediction_proba_1'] = merged_data[proba_cols].mean(axis=1)
    
    

    # Apply threshold using average probability
    mean_threshold = np.mean(optimal_thresholds)
    merged_data['prediction'] = np.where(
        merged_data['average_prediction_proba_1'] >= mean_threshold, 1, 0)



    # Save averaged predictions
    avg_file_path = os.path.join(results_path_classifier_avg, f'{classifier_name}_average_predictions_cv_{cv}_{cv_strategy}_{scoring_method}_threshold_{threshold}_{scale_status}_scale.csv')
    merged_data.to_csv(avg_file_path, index=False)
    logging.info(f"Saved averaged predictions to {avg_file_path}")

    # Compute and save performance metrics for the validation cohort.
    val_data = merged_data[merged_data[cohort_col] == validation_label].copy()

    if val_data.empty:
        raise ValueError(
            f"No validation samples found. "
            f"Check cohort_col={cohort_col} and validation_label={validation_label}.")
    
    # If performance_metrics expects a column called prediction_proba_1,
    # create it from the average column
    val_data['prediction_proba_1'] = val_data['average_prediction_proba_1']

    # Performance
    perf_metrics_val = performance_metrics(
    data_set=val_data,
    optimal_threshold=mean_threshold,
    outcome_col=outcome_col)
    
    perf_metrics_file_path = os.path.join(
        results_path_classifier_avg,
        f"{classifier_name}_average_performance_metrics_tp{timepoint}_cv_{cv}_{cv_strategy}_{scoring_method}_threshold_{threshold}_{scale_status}_scale.csv"
    )    
    perf_metrics_val.to_csv(perf_metrics_file_path, index=False)
    logging.info(f"Saved performance metrics for timepoint {timepoint} to {perf_metrics_file_path}")
    
    # Average feature importances
    if classifier_name == "Logistic_Regression":
        if feature_importance_dfs:
            all_importances = pd.concat(feature_importance_dfs, ignore_index=True)

            avg_importances = (
                all_importances
                .groupby('Feature', as_index=False)
                .agg({
                    'Coefficient': 'mean',
                    'Std': 'first'
                })
            )

            avg_importances['Coef_x_Std'] = np.abs(avg_importances['Coefficient'] * avg_importances['Std'])

            avg_importances['Direction'] = np.where(
                avg_importances['Coefficient'] > 0,
                'Positive',
                'Negative'
            )

            if scale_status:
                avg_importances["Abs_Coefficient"] = np.abs(avg_importances["Coefficient"])
                sort_col = "Abs_Coefficient"
                output_cols = [
                    "Feature",
                    "Coefficient",
                    "Abs_Coefficient",
                    "Std",
                    "Coef_x_Std",
                    "Direction",
                ]
            else:
                sort_col = "Coef_x_Std"
                output_cols = [
                    "Feature",
                    "Coefficient",
                    "Std",
                    "Coef_x_Std",
                    "Direction",
                ]
            
            avg_importances["Coefficient"] = avg_importances["Coefficient"].round(3)
            avg_importances["Std"] = avg_importances["Std"].round(3)
            avg_importances["Coef_x_Std"] = avg_importances["Coef_x_Std"].round(3)
            
            if scale_status:
                avg_importances["Abs_Coefficient"] = avg_importances["Abs_Coefficient"].round(3)
            
            avg_importances = avg_importances[output_cols].sort_values(
                by=sort_col,
                ascending=False,
            )

            avg_importance_file_path = os.path.join(
                results_path_classifier_avg,
                f'{classifier_name}_average_feature_importances_cv_{cv}_{cv_strategy}_{scoring_method}_threshold_{threshold}_{scale_status}_scale.csv'
            )

            avg_importances.to_csv(avg_importance_file_path, index=False)
            logging.info(f"Saved averaged feature importances to {avg_importance_file_path}")



def main():
   
    parser = argparse.ArgumentParser(description='Calculate the average performance metrics over different runs.')
    parser.add_argument('--classifier', type=str, required=True, help='Name of the classifier')
    parser.add_argument('--timepoint', type=int, default=1, help='Timepoint used in the analysis')
    parser.add_argument('--outcome_col', type=str, default='2y_ttp', help='Name of the outcome column')    
    parser.add_argument('--cohort_col', type=str, default='cohort', help='Name of the cohort column')
    parser.add_argument('--validation_label', type=str, default='B', help='Label used for the validation cohort')
    parser.add_argument('--results-path', type=str, required=True, help='Path to the directory containing the results')
    parser.add_argument('--cv', type=int, required=True, help='Number of cross-validation folds')
    parser.add_argument('--cv_strategy', type=str, required=True, help='Cross-validation strategy')
    parser.add_argument('--scoring_method', type=str, required=True, help='Scoring method for optimizing the classifier')
    parser.add_argument('--threshold', type=str, required=True, help='Classification threshold')
    parser.add_argument('--rs', type=int, required=True, help='Number of random states')
    parser.add_argument('--scale_status', type=lambda x: x.lower() == 'true', required=True, help='Whether to scale or not')

    
    args = parser.parse_args()
    average_results(
        args.classifier,
        args.timepoint,
        args.outcome_col,
        args.cohort_col,
        args.validation_label,
        args.results_path,
        args.cv,
        args.cv_strategy,
        args.scoring_method,
        args.threshold,
        args.rs,
        args.scale_status
    )

if __name__ == "__main__":
    main()


# To run this script in the terminal, use the following command:
# python analysis/average_runs.py \
#   --classifier Logistic_Regression \
#   --timepoint 1 \
#   --outcome_col 2y_ttp \
#   --cohort_col cohort \
#   --validation_label B \
#   --results-path results/ \
#   --cv 5 \
#   --cv_strategy StratifiedKFold \
#   --scoring_method roc_auc \
#   --threshold Optimal \
#   --rs 10 \
#   --scale_status True
