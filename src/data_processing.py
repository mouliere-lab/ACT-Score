# Author: Parisa Mapar

import logging


def preprocess_data(
    data,
    timepoint=1,
    outcome_col="2y_ttp",
    file_name_col="subject_id",
    cohort_col="cohort",
    training_label="A",
    validation_label="B"
):
    """
    Preprocess the input data by filtering samples and sorting by sample/subject ID.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data table.
    timepoint : int, default=1
        Timepoint to include in the analysis.
    outcome_col : str, default="2y_ttp"
        Name of the outcome column. Expected values are "Yes" and "No".
    file_name_col : str, default="subject_id"
        Name of the column containing sample or subject identifiers.
    cohort_col : str, default="cohort"
        Name of the column containing training/validation cohort labels.
    training_label : str, default="A"
        Label used for the training cohort.
    validation_label : str, default="B"
        Label used for the validation cohort.

    Returns
    -------
    pandas.DataFrame
        Processed data containing only samples with valid outcome labels,
        training/validation cohort labels, and the selected timepoint.
    """
    
    try:
    
        # Keep only samples with available binary outcome labels.
        data = data[data[outcome_col].isin(["Yes", "No"])]
        
        # Filter the data to include only training and validation samples.
        data = data[data[cohort_col].isin([training_label, validation_label])]

        # Keep only samples from the selected timepoint.      
        data = data[data["timepoint"] == timepoint]
        
        # Sort samples for reproducibility.
        data = data.sort_values(by=[file_name_col]).reset_index(drop=True)

        # Return the processed data
        return data
        
    except KeyError as e:
        # Log an error if a required column is missing and re-raise the exception.
        logging.error(f"Required columns not found in the data: {e}")
        raise
