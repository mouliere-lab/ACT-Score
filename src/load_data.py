# Author: Parisa Mapar

import pandas as pd
import logging

def load_data(file_path):

    """
    Load the input feature table from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the input CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded input data.

    Raises
    ------
    Exception
        Re-raises any exception encountered while reading the file.
    """
    
    try:
        # Attempt to read the CSV file into a pandas DataFrame.
        data = pd.read_csv(file_path)
        
        # Return the loaded data if successful.
        return data
        
    except Exception as e:
    
        # Log an error if there is an issue reading the file and re-raise the exception.
        logging.error(f"Error in reading the input file: {e}")
        raise
    


