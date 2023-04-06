import itertools

from Reverse_Simulator.produce_sensitivity_data import get_baseline_classes, build_class_categories, \
    produce_aerosol_category
from multiprocessing.pool import Pool
import numpy as np

def vary_temp():
    # Want this to just get passed in n aerosols and it does every combination of them
    """
    Produces example data points with temp varied from baseline conditions (lowtemp=298,hightemp=350) and writes them to
    a pkl file

    Returns:
        None
    """
    pool = Pool(9)
    high_temp = [np.arange(385,395,0.1),np.arange(370,380,0.1),np.arange(355,365,0.1)]
    ambient_temp = [np.arange(270,280,0.1),np.arange(280,290,0.1),np.arange(290,300,0.1)]
    combos = itertools.product(high_temp,ambient_temp)
    pool.starmap(vary_single_temp,combos)
    print("---------CATEGORY COMPLETE-----------")

def vary_single_temp(combo,combo2):
    """
    Takes in a range for low and high temp values and produces pkl file output

    Args:
        combo (numpy.ndarray): Low temperature range
        combo2 (numpy.ndarray): High temperature range

    Returns:
        None
    """
    baseline = get_baseline_classes()
    low_rh = np.arange(15,25,0.1)
    high_rh = np.arange(85,95,0.1)
    conditions = {'high_temp':combo,'ambient_temp':combo2,'low_rh':low_rh,'high_rh':high_rh}
    build_class_categories(baseline,mode="Time Averaged",suffix="_test_lowtemp" + str(max(combo)) + "_hightemp" + str(max(combo2)),conditions=conditions)

