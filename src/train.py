# Author: Parisa Mapar

import logging
import warnings

from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Setting up logging configuration for displaying messages with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings for cleaner command-line output.
warnings.filterwarnings('ignore')


def train_classifier(
    train_set,
    features,
    target_train,
    classifier_info,
    classifier_name,
    cv_strategy,
    scoring_method,
    nr_jobs,
    cv_nr,
    scale,
):
    """
    Train a classifier, perform hyperparameter optimization, and tune the classification threshold.

    Parameters
    ----------
    train_set : pandas.DataFrame
        Training data containing the selected features.
    features : list of str
        Feature names used for model training.
    target_train : array-like
        Binary training labels.
    classifier_info : dict
        Dictionary containing the classifier object and hyperparameter grid.
    classifier_name : str
        Name of the classifier.
    cv_strategy : str
        Cross-validation strategy. Supported values are "KFold" and "StratifiedKFold".
    scoring_method : str
        Scoring metric used for hyperparameter optimization.
    nr_jobs : int
        Number of parallel jobs.
    cv_nr : int
        Number of cross-validation folds.
    scale : bool
        Whether to apply standard scaling before model training.

    Returns
    -------
    best_model : sklearn estimator
        Best model selected by GridSearchCV and refit on the full training set.
    optimal_threshold : float
        Classification threshold optimized using balanced accuracy.
    """
    
    try:

        # Define cross-validation strategy - use StratifiedKFold for imbalanced classes

        if cv_strategy == "KFold":
            cv = KFold(n_splits=cv_nr)
        elif cv_strategy == "StratifiedKFold":
            cv = StratifiedKFold(n_splits=cv_nr)
        else:
            raise ValueError("cv_strategy must be 'KFold' or 'StratifiedKFold'")    
            
            
        # Apply scaling if requested.    
        scaler = StandardScaler() if scale else "passthrough"

        pipe = Pipeline([
            ("scaler", scaler),
            ("clf", classifier_info["model"])
        ])
        
        # Prefix classifier hyperparameters with "clf__" so GridSearchCV can tune them inside the pipeline.
        if isinstance(classifier_info["params"], list):
            param_grid = []
            for grid in classifier_info["params"]:
                param_grid.append({f"clf__{k}": v for k, v in grid.items()})
        else:
            param_grid = {f"clf__{k}": v for k, v in classifier_info["params"].items()}

        # Optimize model hyperparameters.
        optimized_classifier = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            refit=True,
            scoring=scoring_method,
            cv=cv,
            verbose=True,
            n_jobs=nr_jobs
        )
        
        optimized_classifier.fit(train_set[features], target_train)
        
        # GridSearchCV refits the best model on the full training set when refit=True.
        best_model = optimized_classifier.best_estimator_
        
        # Log the best parameters found by GridSearchCV
        logging.info(f"{classifier_name} best parameters: {optimized_classifier.best_params_}")

        
        # Optimize the classification threshold using training data only.
        classifier_tuned = TunedThresholdClassifierCV(
            estimator=best_model,
            cv=cv,
            scoring="balanced_accuracy"
        )
        classifier_tuned.fit(train_set[features], target_train)

        logging.info(f"Optimal threshold: {classifier_tuned.best_threshold_:.3f}")

            
        return best_model, classifier_tuned.best_threshold_
            
    except Exception as e:
    
        # Log any error that occurs during training and re-raise the exception for further handling
        logging.error(f"Error in training classifier: {e}")
        raise

