# Author: Parisa Mapar

import warnings
warnings.filterwarnings("ignore")

import logging
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.load_data import load_data

def load_feature_names(feature_file, feature_col="feature"):
    """
    Load feature names from a CSV file.

    Parameters
    ----------
    feature_file : str
        Path to CSV file containing feature names.
    feature_col : str, default="feature"
        Name of the column containing feature names.

    Returns
    -------
    list of str
        Feature names.
    """

    feature_df = pd.read_csv(feature_file)

    if feature_col not in feature_df.columns:
        raise ValueError(f"Feature column '{feature_col}' not found in {feature_file}")

    return feature_df[feature_col].dropna().tolist()


def run_elastic_net_feature_selection(
    output_prefix,
    input_file,
    feature_file,
    output_dir,
    min_times_selected,
    feature_col="feature",
    random_states=10,
    timepoint=1,
    outcome_col="2y_ttp",
    cohort_col="cohort",
    training_label="A",
    validation_label="B",
    n_jobs=16,
):
    """
    Perform Elastic Net-based feature selection using training data only.

    The procedure is repeated across multiple random seeds. Features with
    non-zero logistic regression coefficients are counted across runs.

    Parameters
    ----------
    output_prefix : str
        Prefix used for output file names, for example "LIONHEART" or "DELFI".
    input_file : str
        Path to input CSV file containing features and metadata.
    feature_file : str
        Path to CSV file containing feature names.
    output_dir : str
        Directory where output files will be saved.
    min_times_selected : int
        Minimum number of random-seed runs in which a feature must be selected.    
    feature_col : str, default="feature"
        Name of the column containing feature names in the feature file.    
    random_states : int, default=10
        Number of random seeds to use.
    timepoint : int, default=1
        Timepoint to include in the analysis.
    outcome_col : str, default="2y_ttp"
        Name of the outcome column. Expected values are "Yes" and "No".
    cohort_col : str, default="cohort"
        Name of the column containing training/validation cohort labels.
    training_label : str, default="A"
        Label used for the training cohort.
    validation_label : str, default="B"
        Label used for the validation cohort.    
    n_jobs : int, default=16
        Number of parallel jobs for GridSearchCV.

    Returns
    -------
    None
        Saves run-level results, selected-feature counts, filtered selected features,
        and all selected features across random states.
    """
    
    output_dir = output_dir.rstrip("/") + "/feature_selection"
    os.makedirs(output_dir, exist_ok=True)

    data = load_data(input_file)

    # Keep only samples with available binary outcome labels.
    data = data[data[outcome_col].isin(["Yes", "No"])]
        
    # Filter the data to include only training and validation samples.
    data = data[data[cohort_col].isin([training_label, validation_label])]

    # Keep only samples from the selected timepoint.      
    data = data[data["timepoint"] == timepoint]

    feature_names = load_feature_names(feature_file=feature_file, feature_col=feature_col)

    missing_features = [feature for feature in feature_names if feature not in data.columns]

    if missing_features:
        raise ValueError(
            f"{len(missing_features)} features from the feature file are missing "
            f"in the input data. First missing features: {missing_features[:10]}"
        )

    train_df = data[data[cohort_col] == training_label].copy()

    if train_df.empty:
        raise ValueError(
            f"No training samples found after filtering. "
            f"Check timepoint={timepoint}, cohort_col={cohort_col}, "
            f"training_label={training_label}, and outcome_col={outcome_col}.")

    y_train = np.where(train_df[outcome_col] == "No", 1, 0)
    x_train = train_df[feature_names]

    param_grid = {
        "clf__C": np.logspace(-2, 1, 20),
        "clf__l1_ratio": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    }

    selection_counter = Counter()
    all_run_results = []
    all_selected_features = []

    for random_state in range(1, random_states + 1):
        logging.info(
            f"Running {output_prefix} Elastic Net feature selection "
            f"with random_state={random_state}"
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=20000,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ])

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state)

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=n_jobs,
            refit=True)

        grid.fit(x_train, y_train)

        best_pipeline = grid.best_estimator_
        coefficients = best_pipeline.named_steps["clf"].coef_.ravel()

        coefficient_table = pd.DataFrame({
            "feature": x_train.columns,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }).sort_values("abs_coefficient", ascending=False)

        selected_features = coefficient_table[coefficient_table["coefficient"] != 0].copy()

        selected_features["random_state"] = random_state

        selection_counter.update(selected_features["feature"].tolist())
        all_selected_features.append(selected_features)

        all_run_results.append({
            "random_state": random_state,
            "best_C": grid.best_params_["clf__C"],
            "best_l1_ratio": grid.best_params_["clf__l1_ratio"],
            "best_cv_roc_auc": grid.best_score_,
            "n_nonzero_features": len(selected_features)})

        logging.info(
            f"random_state={random_state}, "
            f"best_cv_roc_auc={grid.best_score_:.3f}, "
            f"n_nonzero_features={len(selected_features)}")

    run_results_df = pd.DataFrame(all_run_results)

    selection_summary = (pd.DataFrame(selection_counter.items(), columns=["feature", "times_selected"])
        .sort_values(["times_selected", "feature"], ascending=[False, True]).reset_index(drop=True))

    selected_features_min_count = selection_summary[selection_summary["times_selected"] >= min_times_selected].copy()

    if all_selected_features:
        all_selected_features_df = pd.concat(all_selected_features, ignore_index=True)
    else:
        all_selected_features_df = pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient", "random_state"])

    run_results_path = os.path.join(output_dir, f"{output_prefix}_feature_selection_run_results.csv")
    selection_counts_path = os.path.join(output_dir, f"{output_prefix}_selected_feature_counts.csv")
    selected_min_count_path = os.path.join(output_dir, f"{output_prefix}_selected_features_min{min_times_selected}.csv")
    all_selected_path = os.path.join(output_dir, f"{output_prefix}_all_selected_features_by_random_state.csv")

    run_results_df.to_csv(run_results_path, index=False)
    selection_summary.to_csv(selection_counts_path, index=False)
    selected_features_min_count.to_csv(selected_min_count_path, index=False)
    all_selected_features_df.to_csv(all_selected_path, index=False)

    logging.info(f"Saved run results to {run_results_path}")
    logging.info(f"Saved feature selection counts to {selection_counts_path}")
    logging.info(
        f"Saved features selected at least {min_times_selected} times "
        f"to {selected_min_count_path}")
    logging.info(f"Saved all selected features by random state to {all_selected_path}")
