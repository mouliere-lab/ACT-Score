# Author: Parisa Mapar

import logging


def test_classifier(classifier, test_set, features):

    """
    Generate class predictions and positive-class probabilities for a fitted classifier.

    Parameters
    ----------
    classifier : sklearn estimator or Pipeline
        Fitted classifier.
    test_set : pandas.DataFrame
        Data containing the selected feature columns.
    features : list of str
        Feature names used for prediction.

    Returns
    -------
    y_test_pred : array-like
        Predicted class labels.
    y_test_proba : array-like
        Predicted probabilities for the positive class.
    """

    try:
        # Predict class labels
        y_test_pred = classifier.predict(test_set[features])
        
        # Predict probability of the positive class.
        y_test_proba = classifier.predict_proba(test_set[features])[:, 1]
        
        return y_test_pred, y_test_proba
        
    except Exception as e:
        # Log any errors encountered during prediction
        logging.error(f"Error in testing classifier: {e}")
        raise
        
