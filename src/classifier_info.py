# Author: Parisa Mapar

import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn import svm



def get_classifier_info(classifier_name, nr_jobs):

    """
    Return the classifier object and hyperparameter grid for model training.

    Parameters
    ----------
    classifier_name : str
        Name of the classifier.
    nr_jobs : int
        Number of parallel jobs used by classifiers that support parallelization.

    Returns
    -------
    dict
        Dictionary containing the classifier object and hyperparameter grid.

    Raises
    ------
    ValueError
        If the classifier name is not supported.
    """

    # Dictionary containing models and their respective hyperparameter grids for tuning
    classifiers = {
    
    # Random Forest model with a specified number of parallel jobs
        'Random_Forest': {'model': RandomForestClassifier(random_state=42, n_jobs=nr_jobs),
        'params': {
            'n_estimators': [30, 50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced'],
            'oob_score':[False],
            'max_features':['sqrt', None],
            'criterion':['gini', 'entropy'],
            'max_samples':[None, 0.5, 0.75, 1]
        }
    },

 

    # Logistic Regression model with a maximum iteration limit and parallel jobs
    'Logistic_Regression': {
        'model': LogisticRegression(max_iter=10000, random_state=42),
        'params': [
            {
                'penalty': ['l1'],
                'C': np.logspace(-2, 2, 20),
                'fit_intercept': [True, False],
                'class_weight': [None, 'balanced'],
                'solver': ['liblinear', 'saga']
            },
            {
                'penalty': ['l2'],
                'C': np.logspace(-2, 2, 20),
                'fit_intercept': [True, False],
                'class_weight': [None, 'balanced'],
                'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga', 'newton-cholesky']
            },
            {
                'penalty': ['elasticnet'],
                'C': np.logspace(-2, 2, 20),
                'fit_intercept': [True, False],
                'class_weight': [None, 'balanced'],
                'solver': ['saga'],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            {
                'penalty': [None],
                'fit_intercept': [True, False],
                'class_weight': [None, 'balanced'],
                'solver': ['lbfgs', 'newton-cg', 'sag', 'saga', 'newton-cholesky']
            }
        ]
    },
    
    # Gaussian Naive Bayes model
    'Gaussian_Naive_Bayes': {
        'model': GaussianNB(),
        'params':{
            'var_smoothing': [10**-i for i in range(2, 16)]
     }
    },
    
    # Support Vector Classifier with specified random state
    'SVM': {
        'model': svm.SVC(probability=True, random_state=42),
        'params':{
                'C': [0.1, 1, 10, 100,1000],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto',1, 0.1, 0.01, 0.001, 0.0001],
                'class_weight': [None, 'balanced'],
                'probability':[True],
                'degree' : [0, 1, 2, 3, 4, 5]
        }
    },

    'Gradient_Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params':{
            'n_estimators':[16, 32, 50, 64, 100, 200],
            'max_depth':np.arange(5,30,3), 
            'min_samples_split':np.linspace(0.1, 1.0, 10, endpoint=True),
            'min_samples_leaf':np.linspace(0.1, 0.5, 5, endpoint=True),
            'subsample':[0.6,0.7,0.8,0.9],
            'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01]
        }
    },
    
  
    'EasyEnsemble': {
        'model': EasyEnsembleClassifier(random_state=42, n_jobs=nr_jobs),
        'params':{
            'n_estimators': [10, 20, 30]
        }
    }}
    
    # Check if the provided classifier name exists in the dictionary
    if classifier_name not in classifiers:
        supported_classifiers = ", ".join(classifiers.keys())
        raise ValueError(
            f"Classifier '{classifier_name}' is not supported. "
            f"Supported classifiers are: {supported_classifiers}."
        )
    
    # Return the specified classifier's model and hyperparameter grid
    return classifiers[classifier_name]
    