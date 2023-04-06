import os

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.stats import norm
import pandas as pd

from Example_Graphs import species_rh_test
from Forward_Simulator import construct_realistic_output

def check_opc_modality(opc_output):
    """
    Check modality of a binned opc output

    Args:
        opc_output (pandas.Dataframe): Output from a multi-modal aerosol distribution

    Returns:
        int: Most likely modality
    """
    opc_output = opc_output.reshape((-1,1))
    possible_components = {"n_components": range(1, 7),}
    search = GridSearchCV(GaussianMixture(), param_grid=possible_components, scoring=bic)
    search.fit(opc_output)
    bic_scores = list(search.cv_results_['mean_test_score'])
    most_likely = bic_scores.index(min(bic_scores))
    # Might have to bias this somehow
    return most_likely+1

def bic(estimator, data):
    return estimator.bic(data)

# Look into this further
def get_bimodal_bin_weights(opc_output):
    """
    Get info on modes of binned opc output

    Args:
        opc_output (pandas.Dataframe): Output from a multi-modal aerosol distribution

    Returns:
        Dict[str,float]: Dictionary of opc mode info
    """
    gmm = GaussianMixture(n_components=2,covariance_type='full')
    gmm.fit(opc_output.reshape(-1, 1))
    means = gmm.means_.flatten()

    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_
    converged = gmm.converged_
    if converged:
        bool_converged = 1
    else:
        bool_converged = 0
    x = np.linspace(min(opc_output),max(opc_output), len(opc_output))

    cat_weights = []
    for mean, covariance, weight in zip(means, covariances, weights):
        cat_weights.append(weight * norm.pdf(x, mean, np.sqrt(covariance)))
    return cat_weights,[covariances,weights,bool_converged]

# Start by assuming there are two components
def split_bimodal_dist(opc_output):
    """
    Construct dictionary of parameters from a bimodal OPC output

    Args:
        opc_output (Dict[str,float]): Output from a multi-modal aerosol distribution

    Returns:
        Dict[str,float]: Dictionary of parameters
    """
    # Normalized Usable Output
    opc_output.pop("rh")
    opc_output.pop('temp')
    opc_output = np.asarray(list(opc_output.values()))

    # Useless when normalizing outputs
    bimodal_params = {"total_particles":sum(opc_output)}

    weights,modal_params = get_bimodal_bin_weights(opc_output)


    weights = normalize_weights(weights)
    first_mode = []
    second_mode = []
    for w0,w1,bin_count in zip(weights[0],weights[1],opc_output):
        first_mode.append(w0*bin_count)
        second_mode.append(w1*bin_count)

    # Get median, mean, st_dev, and range
    if sum(first_mode) == 0 or sum(second_mode) == 0:
        bimodal_params['mode2'] = 0
        bimodal_params['mode_maxes'] = 0.
        bimodal_params['maxes_diff'] = 0.
        bimodal_params['Weight_dif'] = abs(max(modal_params[1]))


    else:
        maxes = get_bimodal_max([first_mode, second_mode])
        bimodal_params['mode2'] = 1
        bimodal_params['mode_maxes'] = max(maxes) - min(maxes)
        bimodal_params['maxes_diff'] = abs(max(maxes) - min(maxes))
        bimodal_params['Weight_dif'] = abs(max(modal_params[1]) - min(modal_params[1]))

    return bimodal_params

def normalize_weights(weights):
    """
    Normalize bin mode weights

    Args:
        weights List[]: Bin weights for two modes

    Returns:
        List[]: Normalized bin weights
    """
    # Want the two probabilities to add to one

    # Add the two probs
    # Divide each weight by that number
    for i in range(len(weights[0])):
        tot = weights[0][i] + weights[1][i]
        weights[0][i] /= tot
        weights[1][i] /= tot
    return weights

# Could try to split the counts, both bin to bin and within a single bin
def get_bimodal_means(mode_weights):
    """
    Get mean diameters from bimodal OPC outputs

    Args:
        mode_weights (List[float]): Weight of single modes in bimodal flow

    Returns:
        List[float]: Mean diameters
    """
    means = []
    bin_diameters = species_rh_test.get_n3_bin_size()
    midpoints = [bd[1] for bd in bin_diameters]
    for mode in mode_weights:
        total = sum(mode)
        summed = 0.
        for weight,diameter in zip(mode,midpoints):
            summed+=weight*diameter
        means.append(summed/total)
    return means

def get_bimodal_max(mode_weights):
    """
    Get max bin for different OPC output modes

    Args:
    mode_weights List[float]: List of single mode OPC weights

    Returns:
        List[int]: Max bins
    """
    bin_diameters = species_rh_test.get_n3_bin_size()
    midpoints = [bd[1] for bd in bin_diameters]
    max_bin = []
    for mode in mode_weights:
        #print(mode.index(max(mode)))
        #print(len(mode))
        #print(mode)
        #print(len(midpoints))
        max_bin.append(midpoints[mode.index(max(mode))])
    return max_bin

def get_mode_ranges(modes):
    """
    Get indexes of mode ranges

    Args:
        modes (List[List[int]]): Single flow OPC outputs

    Returns:
        List[List[int]]: Ranges of opc output modes
    """
    ranges = []
    for mode in modes:
        start_index = -10
        stop_index = -10
        # Loop through bins, start when a value is non-zero, stop when a value is zero
        for i in range(0,len(mode)):
            if mode[i] != 0 and start_index==-10:
                start_index = i
            elif mode == 0 and start_index != -10:
                stop_index = 1
            break
        ranges.append([start_index,stop_index])
    return ranges

def get_multi_component_stats(baseline,first=False, **opc_output):
    """
    Get stats for multi-component data processing

    Args:
        first (bool): Boolean on if this is the baseline flow
        baseline (Dict[str,float]): Baseline opc output
        opc_output (Dict[str,float]): Experimental opc output

    Returns:
        Dict[str,float]
    """

    if first:
        return split_bimodal_dist(baseline)
    else:
        return get_relative_bimodal_dist(baseline,opc_output['opc_output'])

def get_relative_bimodal_dist(baseline, opc_out):
    """
    Makes all parameters relative to a baseline value

    Args:
        baseline (Dict[str,float]): Baseline opc output
        opc_out (Dict[str,float]): Experimental opc output

    Returns:
        Dict[str,float]: Experimental output relative to baseline
    """
    test_out = split_bimodal_dist(opc_out)
    #print("BASELINE")
    #print(baseline)
    #print("OPC OUT")
    #print(test)
    for key in baseline.keys():
        if "mode" in key:
            continue
        if key in test_out.keys():
            if baseline[key] == 0:
                if test_out[key] ==0:
                    continue
                else:
                    test_out[key] = 1
            else:
                test_out[key] = (test_out[key]-baseline[key])/baseline[key]
    return test_out

def convert_time_averaged(folder,names, mode):
    """
    Convert time-averaged data to other modes

    Args:
        folder (str): Folder with time-averaged data
        names (List[str]): Names of files in folder
        mode (str): Data processing mode

    Returns:
        None
    """
    cwd = os.path.dirname(os.path.abspath(os.getcwd()))
    # Read in data from each json as a dataframe
    data = None
    classes = None
    for name in names:
        data_point = pd.read_pickle(
            cwd + "/OpcSimResearch/Aerosol_Category_Outputs/" + folder + "/" + name + "/" + name + "_category_examples.pkl")
        d_list = []
        for d in data_point:
            if mode == "Multi-Component":
                average_size =float(construct_realistic_output.get_average_particle_size(d.iloc[0].to_dict()))
                mode_descriptors = get_multi_component_stats(baseline=d.iloc[0].to_dict(), first=True)
                mode_descriptors['average_size'] = average_size
                output = [get_multi_component_stats(mode_descriptors, False, opc_output=d.iloc[1].to_dict()),
                          get_multi_component_stats(mode_descriptors, False, opc_output=d.iloc[2].to_dict()),
                          get_multi_component_stats(mode_descriptors, False, opc_output=d.iloc[3].to_dict()),
                          get_multi_component_stats(mode_descriptors, False, opc_output=d.iloc[4].to_dict())]
                mc_data = pd.DataFrame(output)
                print(mc_data)
                d_list.append(mc_data)
        # Plot this data to look for differences
        data_point = d_list
        if data is None:
            data = data_point
        else:
            data = np.concatenate((data, data_point))
        if classes is None:
            classes = np.full(len(data_point), name)
        else:
            classes = np.append(classes, np.full(len(data_point), name))
    return data, classes