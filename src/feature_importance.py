# Author: Parisa Mapar

import numpy as np
import pandas as pd
import os
import logging

def calculate_feature_importances(cf, classifier_name):

    """
    Extract logistic regression coefficients from a fitted pipeline.

    Parameters
    ----------
    cf : sklearn Pipeline
        Fitted model pipeline.
    classifier_name : str
        Name of the classifier.

    Returns
    -------
    numpy.ndarray
        Logistic regression coefficients.
    """

    try:
    
        if classifier_name != "Logistic_Regression":
            raise ValueError(
                "Feature importance calculation is only implemented for Logistic_Regression."
            )
        
        
        # Use the coefficients of the best estimator for logistic regression
        imps = cf.named_steps["clf"].coef_[0]
            
        return imps
        
    except Exception as e:
        # Log any errors encountered
        logging.error(f"Error in calculating feature importances: {e}")
        raise
    
    
def save_feature_importances(importances, features, train_set,results_path_classifier, classifier_name, title, scale_status):

    """
    Save logistic regression coefficients to a CSV file.

    Parameters
    ----------
    importances : array-like
        Logistic regression coefficients.
    features : list of str
        Feature names used by the model.
    train_set : pandas.DataFrame
        Training dataset.
    results_path_classifier : str
        Directory where output files will be saved.
    classifier_name : str
        Name of the classifier.
    title : str
        Suffix used in the output file name.
    scale_status : bool
        Whether feature scaling was applied during model training.

    Returns
    -------
    None
    """
    
    try:
        
        if classifier_name == 'Logistic_Regression':
            imp = pd.DataFrame({
                "Feature": features,
                "Coefficient": importances,
                "Std": train_set[features].std().values})
            
            imp["Coef_x_Std"] = np.abs(imp["Coefficient"] * imp["Std"])
            imp["Direction"] = np.where(imp["Coefficient"] > 0, "Positive", "Negative")
            
            # If features were scaled, rank by absolute coefficient.
            # If features were not scaled, rank by coefficient multiplied by feature standard deviation.
            
            if scale_status:
                imp["Abs_Coefficient"] = np.abs(imp["Coefficient"])
                imp = imp.sort_values("Abs_Coefficient", ascending=False)

                imp["Abs_Coefficient"] = imp["Abs_Coefficient"].round(3)
            else:
                imp = imp.sort_values("Coef_x_Std", ascending=False)
                
            imp["Coefficient"] = imp["Coefficient"].round(3)
            imp["Std"] = imp["Std"].round(3)
            imp["Coef_x_Std"] = imp["Coef_x_Std"].round(3)
            
            imp.to_csv(os.path.join(results_path_classifier, f'{classifier_name}_feature_importances_{title}.csv'),index=False)

 
    except Exception as e:
        # Log any errors encountered
        logging.error(f"Error in saving feature importances: {e}")
        raise

