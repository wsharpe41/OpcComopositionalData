# Should be able to get experimental conditions from a csv for the simulator optimizer
"""

This script requires both pandas and OPCSim be installed on your system

"""
import pandas as pd
import numpy as np
from Example_Graphs import species_rh_test
from Reverse_Simulator import reverse_sim_utils
pd.set_option('display.max_columns', None)

def simulate_under_given_conditions(conditions,aerosol,experimental_info,td_info,mode,initial_size,initial_params=None):
    """
    Simulate the response of an aerosol to different relative humidity and temperatures with three different output formats

    Args:
        conditions (pandas.DataFrame): Dataframe of RH and temperatures (K) that will be used for experiments
        aerosol (AerosolClass): AerosolClass object that will be used for experiments
        experimental_info (Dict[str,float]): Dictionary containing experiment length and sampling rate
        td_info (Dict[str,float]): Dictionary containing TD residence time (s), time step (s), and number of particles (#/cc)
        mode (str): One of three output modes: Full Output, Time Averaged, or Median Diameter
        initial_size (float): Initial median diameter of aerosol particles
        initial_params: Additional parameter to be passed depending on mode

    Returns:
        pandas.Dataframe: Dataframe containing results of all experiments described in conditions DataFrame
    """
    all_experiments = []
    for index, row in conditions.iterrows():
        experiment_results = aerosol.simulate_constant_experiment(experimental_length=experimental_info['length'],sampling_rate=experimental_info['sr'],
                                                                  residence_time=td_info['res'],time_step=td_info['ts'],rh=row['rh'],
                                                                  temperature=row['temp'],number_of_particles=td_info['np'],randomness=True)
        if mode == 'Full Output':
            all_experiments.append(normalize_experiment_for_number_of_particles(experiment_results).to_dict())
        else:
            non_norm_output = average_simulated_experiments(experiment_results)
            averaged_output = average_simulated_experiments(normalize_experiment_for_number_of_particles(experiment_results))
            rh = averaged_output['rh']
            temp = averaged_output['temp']
            if mode == 'Time Averaged':
                all_experiments.append(averaged_output)
            if mode == 'Relative_Time_Averaged':
                if initial_size == -99:
                    initial_size = averaged_output
                    return initial_size
                else:
                    all_experiments.append(get_relative_bins(initial_size,averaged_output))
            if mode == 'Median Diameter' or mode == "Parameter Estimation" or mode == "Multi-Component":
                average_size =float(get_average_particle_size(averaged_output))
                if mode == 'Multi-Component':
                    # Just these plus average size
                    if initial_size == -99:
                        mode_descriptors = reverse_sim_utils.get_multi_component_stats(baseline=non_norm_output,first=True)
                        mode_descriptors['size_change'] = average_size
                        return average_size,mode_descriptors
                    else:
                        mode_descriptors = (reverse_sim_utils.get_multi_component_stats(initial_params,opc_output=non_norm_output,first=False))
                        if initial_params['size_change'] == 0:
                            if average_size == 0:
                                mode_descriptors['size_change'] = 0
                            else:
                                mode_descriptors['size_change'] = 1
                        mode_descriptors['size_change'] = (average_size-initial_params['size_change'])/initial_params['size_change']
                        mode_descriptors['average_size'] = average_size
                        all_experiments.append(mode_descriptors)
                    continue
                # Get percent change
                if initial_size ==-99:
                    return float(average_size)
                if initial_size == 0:
                    if average_size ==0:
                        size_change = 0.
                    else:
                        size_change = 1.
                else:
                    size_change = (initial_size - average_size)/initial_size
                all_experiments.append({'average_size':average_size,'rh':rh,'temp':temp, 'size_change':size_change})
    data = pd.DataFrame(all_experiments)
    data.replace(np.nan,0)
    return data

def average_simulated_experiments(simulated_output):
    """
    Averages time series opc bin counts

    Args:
        simulated_output (pandas.Dataframe): DataFrame of OPC Bin counts for a given experiment

    Returns:
        Dict[str,float]: Dictionary containing the average value of each OPC bin
    """

    # RH and Temp column will just become first value of column
    # Bin columns will become average of their values
    bin_simulated_output = simulated_output.drop(["rh", "temp"], axis=1)
    averaged_values = (bin_simulated_output.mean(axis=0)).to_dict()
    # Just averaging the whole thing gave sloppy values for temp and rh
    averaged_values['rh'] = simulated_output['rh'].iloc[0]
    averaged_values['temp'] = simulated_output['temp'].iloc[0]
    return averaged_values

def get_average_particle_size(time_averaged_output):
    """
    Get average particle size from OPC bin outputs (currently setup for AlphaSense-N3)

    Args:
        time_averaged_output (Dict[str,float]: Dictionary containing average bin counts for each bin in an OPC

    Returns:
        float: Average particle diameter
    """
    # Take in all of the bin counts, particle diameters associated with each bin
    bin_diameters = species_rh_test.get_n3_bin_size()
    # For each bin take the midpoint
    # This assumes that the particles are normally distributed within each bin
    midpoints = [bd[1] for bd in bin_diameters]
    # Weight each diameter according to bin concentration and output
    time_averaged_output.pop('rh')
    time_averaged_output.pop('temp')
    # Multiply each bin diameter by it's concentration then divide by the sum of the concentrations
    all_particles = sum(time_averaged_output.values())
    averaged_list = list(time_averaged_output.values())
    summed_count = 0.0
    for i in range(0, len(midpoints)):
        summed_count+=midpoints[i]*averaged_list[i]
    if all_particles == 0:
        return 0
    average_size = summed_count/all_particles
    return average_size

def normalize_experiment_for_number_of_particles(simulated_output):
    """
    Normalizes OPC bin counts to abstract away flow particle density

    Args:
        simulated_output (pandas.Dataframe): OPC bin counts

    Returns:
        pandas.Dataframe: Normalized OPC bin counts
    """
    # Take the 'bin' columns, get the biggest value of all of them and divide the other values by that
    # Simulated Output is a pd dataframe
    non_bin_info = simulated_output.loc[:,['rh','temp']]
    bin_simulated_output = simulated_output.drop(["rh","temp"],axis=1)
    max_value = bin_simulated_output.max().max()
    if max_value == 0:
        normalized_output = bin_simulated_output
    else:
        normalized_output = bin_simulated_output/max_value
    normalized_output['rh'] = non_bin_info['rh']
    normalized_output['temp'] = non_bin_info['temp']
    return normalized_output

def get_relative_bins(initial_opc,final_opc):
    """
    Get Relative change in bin count for a given OPC output given a reference opc count

    Args:
        :param pandas.Dataframe initial_opc: Reference OPC bin counts
        :param pandas.Dataframe final_opc: Experimental OPC bin counts

    Returns:
        pandas.Dataframe: Relative bin counts
    """
    # For each bin do percent change
    relative_change = {}
    for bin_count in initial_opc:
        if bin_count == "rh" or bin_count == "temp":
            continue
        if bin_count in final_opc:
            start = initial_opc[bin_count]
            end = final_opc[bin_count]
            if start == 0:
                if end == 0:
                    relative_change[bin_count] = 0
                else:
                    relative_change[bin_count] = 1.
            else:
                relative_change[bin_count] = (end-start)/start
    return relative_change