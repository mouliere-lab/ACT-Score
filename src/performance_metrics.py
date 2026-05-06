# Author: Parisa Mapar

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

def performance_metrics(data_set, optimal_threshold, outcome_col="2Y_PFS"):

    """
    Calculate classification performance metrics.

    Parameters
    ----------
    data_set : pandas.DataFrame
        Dataset containing outcome labels and prediction probabilities.
    optimal_threshold : float
        Threshold used to convert predicted probabilities into binary predictions.
    outcome_col : str, default="2Y_PFS"
        Name of the outcome column. Expected values are "Yes" and "No".

    Returns
    -------
    pandas.DataFrame
        DataFrame containing performance metrics.
    """

    # Define the columns for the results DataFrame to store various performance metrics.
    results_columns = [
        "Precision (Positive Predictive Value)",
        "Negative Predictive Value",
        "Sensitivity",
        "Specificity",
        "Accuracy",
        "Balanced Accuracy",
        "F1 Score",
        "AUC",
        "Optimal threshold",
    ]
    
    # Initialize an empty DataFrame with the specified columns.
    results = pd.DataFrame(columns=results_columns)

    try:
    
        df = data_set.copy()
    
        # Encode labels: 1 = progressor/outcome "No", 0 = non-progressor/outcome "Yes".
        target_labels = np.where(df[outcome_col] == "No", 1, 0)
        
        y_score = df['prediction_proba_1'].to_numpy()
        y_pred = (y_score >= optimal_threshold).astype(int)
        
        # Calculate the false positive rate and true positive rate for the ROC curve.
        fpr, tpr, _ = roc_curve(target_labels, y_score)

        # Compute the Area Under the Curve (AUC) from the ROC curve.
        auc_score = auc(fpr, tpr)

        # Calculate precision (positive predictive value) with zero_division set to 0 to handle divisions by zero.
        precision = precision_score(target_labels, y_pred, zero_division=0)
        
        # Get the confusion matrix and extract true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp).
        tn, fp, fn, tp = confusion_matrix(target_labels, y_pred).ravel()
        
        # Calculate the negative predictive value (NPV).
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Calculate sensitivity (recall).
        sensitivity = recall_score(target_labels, y_pred, zero_division=0)
        
        # Calculate specificity (true negative rate).
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate balanced accuracy
        balanced_accuracy = balanced_accuracy_score(target_labels, y_pred)
        
        # Calculate accuracy
        accuracy = accuracy_score(target_labels, y_pred)
        
        # Calculate the F1 score.
        f1 = f1_score(target_labels, y_pred , zero_division=0)
        
        # Store all calculated metrics in the results DataFrame, rounding them to two decimal places.
        results.loc[0, 'Precision (Positive Predictive Value)'] = round(precision, 2)
        results.loc[0, 'Negative Predictive Value'] = round(npv, 2)
        results.loc[0, 'Sensitivity'] = round(sensitivity, 2)
        results.loc[0, 'Specificity'] = round(specificity, 2)
        results.loc[0, 'Balanced Accuracy'] = round(balanced_accuracy, 2)
        results.loc[0, 'Accuracy'] = round(accuracy, 2)
        results.loc[0, 'F1 Score'] = round(f1, 2)
        results.loc[0, 'AUC'] = round(auc_score, 2)

        results.loc[0, 'Optimal threshold'] = optimal_threshold
        
        # Return the DataFrame containing the performance metrics.
        return results

    except Exception as e:
    
        # Log any error encountered during the metric calculation and re-raise the exception.
        logging.error(f"Error in calculating performance metrics: {e}")
        raise
        
    

