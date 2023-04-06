import numpy as np

from Reverse_Simulator.produce_sensitivity_data import get_baseline_classes,build_class_categories
from multiprocessing.pool import Pool

def vary_flow_gsd():
    """
    Produces example data points with gsd varied from baseline conditions (gsd=1.2-1.6) and writes them to a pkl
    file

    Returns:
        None
    """
    pool = Pool(3)
    gsd_ranges = [np.arange(.25, .75, .01), np.arange(.5, 1.5, .01), np.arange(1., 3., .01)]
    pool.map(single_gsd,gsd_ranges)

def single_gsd(gsd_range):
    """
    Takes in a single gsd range and produces data points

    Args:
        gsd_range (numpy.ndarray): range of possible flow gsd values
    Returns:
        None

    """
    baseline = get_baseline_classes()
    baseline[0].particle_diameter = [gsd_range]
    baseline[1].particle_diameter = [gsd_range]
    baseline[2].particle_diameter = [gsd_range]
    baseline[3].particle_diameter = [gsd_range]
    build_class_categories(aerosols=baseline, suffix="gsd___" + str(max(gsd_range))[0:4],
                           mode='Time Averaged')
    return

