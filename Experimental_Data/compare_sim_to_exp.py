from Experimental_Data import read_in_experimental_data
import numpy as np
from statistics import mean
# Make graph from OPC Sim and plot it on same graph as experimental results

def compare_experiment_to_simulator(sim_output,experimental_path):
    """
    Compare experimentally determined OPC response to simulated response under the same conditions

    Args:
        sim_output (pandas.Dataframe): Simulated outputs
        experimental_path (str): File Path to experimental output

    Returns:
        float:  Euclidean distance between actual and simulated output
    """
    # Dataframe with lots of columns
    experimental_data = read_in_experimental_data.read_in_exp(experimental_path)
    # Sim output is Dataframe with columns for RH, Temp, and bin counts
    common_columns = np.intersect1d(experimental_data.columns, sim_output.columns)
    euc_simil_all = {}
    # Common output should be temperature, bins and RH
    # Might have to clean the data from the experiments
    # Need these to be the same length
    for column in common_columns:
        if "bin" in column:
            experiment_length = (len(experimental_data[column]))
            euc_simil = calc_euclidean(experimental_data[column],sim_output[column])
            euc_simil_all[column] = euc_simil/experiment_length
    print(euc_simil_all)
    print(mean(euc_simil_all.values()))
    # https://tech.gorilla.co/how-can-we-quantify-similarity-between-time-series-ed1d0b633ca0
    return euc_simil_all


def calc_euclidean(experimental, sim):
    """
    Calculate Euclidean similarity

    Args:
        experimental (float): Experimental bin count
        sim (float): Simulated bin count

    Returns:
        float: Euclidean similarity
    """
    return np.sqrt(np.sum((experimental - sim) ** 2))
