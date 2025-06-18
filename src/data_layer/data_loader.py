import pandas as pd
import os

def load_data(data_path="data/Healthcare_Dataset_Preprocessed.csv"):
    """
    Load the dataset from the specified path.
    
    Args:
        data_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data_path = os.path.join(data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    return pd.read_csv(data_path)