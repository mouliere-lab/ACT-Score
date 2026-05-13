# ACT-score

## Overview

ACT score is a machine learning framework for early outcome prediction in large B-cell lymphoma (LBCL), including diffuse large B-cell lymphoma (DLBCL), using plasma cell-free DNA (cfDNA) collected during first-line treatment.

ACT stands for:

- **A**: genomic aberrations, represented by tumor fraction estimated
- **C**: cfDNA fragment composition, including fragment size and genome-wide fragmentation features
- **T**: terminal motif analyses, including fragment-end motif diversity and motif-based scores

The pipeline is designed to integrate genomic and fragmentomic features from a single on-treatment plasma sample, typically collected after one cycle of therapy, to identify patients at higher risk of progression.

In the current manuscript, ACT integrates multiple cfDNA feature classes, including tumor fraction, fragment size distributions, motif-based features, LIONHEART-derived cfDNA tissue/cell-type contribution scores, and DELFI-FTK genome-wide fragmentation features. High-dimensional LIONHEART and DELFI-FTK features are selected using Elastic Net logistic regression within the training cohort only.

The model development framework uses a training cohort and an independent validation cohort. All feature screening, feature selection, hyperparameter tuning, model fitting, and threshold optimization are performed using training data only. The validation cohort is used only for final evaluation.

## Repository structure

```text
ACT-score/
├── analysis/
│   ├── feature_selection.py
│   ├── train_validate.py
│   ├── average_runs.py
│   └── predict_new_samples.py
├── src/
│   ├── classifier_info.py
│   ├── data_processing.py
│   ├── elastic_net_feature_selection.py
│   ├── evaluate.py
│   ├── feature_importance.py
│   ├── load_data.py
│   ├── performance_metrics.py
│   └── train.py
├── scripts/
│   ├── run_feature_selection.sh
│   ├── run_train_validate.sh
│   └── run_predict_new_samples.sh
├── data/
│   ├── input_feature_table.csv
│   └── feature_lists/
│       ├── core_features.csv
│       ├── LIONHEART_features.csv
│       └── DELFI-FTK_features.csv
├── results/
├── requirements.txt
└── README.md
