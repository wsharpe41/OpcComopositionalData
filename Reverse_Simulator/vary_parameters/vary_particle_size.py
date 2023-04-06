from Reverse_Simulator.produce_sensitivity_data import get_baseline_classes, build_class_categories
import numpy as np
from multiprocessing.pool import Pool

def vary_uniform_particle_diameter():
    """
    Produces example data points with particle diameter varied from baseline conditions (.25-10um) and writes them to a pkl
    file

    Returns:
        None
    """
    # Vary particle diameters
    # Vary uniform particle size
    # (.25-1),(.5-2),(1-4)
    pool = Pool(4)
    particle_diameter_range = [np.arange(.25, 1., .01),np.arange(.5, 2., .01),np.arange(1., 4., .01),np.arange(.25,2.25,.01)]
    pool.map(vary_single_uniform_particle_diameter, particle_diameter_range)
    return

def vary_single_uniform_particle_diameter(particle_range):
    """
    Takes in range of a flow's median particle diameter and writes to pkl file

    Args:
        particle_range (numpy.ndarray): Numpy array containing possible particle diameters

    Returns:
        None
    """
    # Vary particle diameters
    # Vary uniform particle size
    # (.25-1),(.5-2),(1-4)
    baseline = get_baseline_classes()
    baseline[0].particle_diameter = [particle_range]
    baseline[1].particle_diameter = [particle_range]
    baseline[2].particle_diameter = [particle_range]
    baseline[3].particle_diameter = [particle_range]
    build_class_categories(aerosols=baseline,suffix="diameter___" + str(max(particle_range))[0:4],mode='Time Averaged')
    # 3 combos
    return
