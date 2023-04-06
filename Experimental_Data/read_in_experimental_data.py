import pandas as pd

def read_in_exp(path):
    """
    Read in OPC bin counts from csv

    Args:
        path (str): csv path

    Returns:
        pandas.Dataframe: OPC Counts
    """
    experimental_data = pd.read_csv(path,header=3)
    return experimental_data
