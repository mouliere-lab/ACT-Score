# Author: Parisa Mapar

import logging


def preprocess_data(
    data,
    timepoint=1,
    outcome_col="2Y_PFS",
    file_name_col="file_name"
):
    """
    Preprocess the input data by filtering samples and sorting by file name.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data table.
    timepoint : int or float, default=1
        Timepoint to include in the analysis.
    file_name_col : str, default="file_name"
        Name of the column containing sample/file identifiers.
    outcome_col : str, default="2Y_PFS"
        Name of the outcome column. Expected values are "Yes" and "No".

    Returns
    -------
    pandas.DataFrame
        Processed data containing only samples with valid outcome labels,
        training/validation split, and the selected timepoint.
    """
    
    try:
    
        # Keep only samples with available binary outcome labels.
        data = data[data[outcome_col].isin(["Yes", "No"])]
        
        # Filter the data to include only training and validation samples.
        data = data[data["split"].isin(["Training", "Validation"])]

        # Keep only samples from the selected timepoint.      
        data = data[data['timepoint'] == 1]
        
        # Sort samples for reproducibility.
        data = data.sort_values(by=[file_name_col]).reset_index(drop=True)

        # Return the processed data
        return data
        
    except KeyError as e:
        # Log an error if a required column is missing and re-raise the exception.
        logging.error(f"Required columns not found in the data: {e}")
        raise
