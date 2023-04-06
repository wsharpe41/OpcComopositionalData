import itertools

from Reverse_Simulator.produce_sensitivity_data import get_baseline_classes, build_class_categories, \
    produce_aerosol_category
from multiprocessing.pool import Pool
import numpy as np

def vary_rh():
    """
    Produces example data points with RH varied from baseline conditions (lowrh=20,highrh=90) and writes them to a pkl
    file

    Returns:
        None
    """
    # Want this to just get passed in n aerosols and it does every combination of them
    pool = Pool(2)
    low_rhs =[np.arange(10,20,0.1),np.arange(20,30,0.1),np.arange(30,40,0.1)]
    high_rhs = [np.arange(60,70,0.1),np.arange(70,80,0.1),np.arange(80,90,0.1)]
    combos = list(itertools.product(low_rhs, high_rhs))
    pool.starmap(vary_single_rh,combos)
    print("---------CATEGORY COMPLETE-----------")

def vary_single_rh(combo,combo2):
    """
    Takes in a range for low and high rh values and produces pkl file output

    Args:
        combo (numpy.ndarray): Low RH range
        combo2 (numpy.ndarray): High RH range

    Returns:
        None
    """
    baseline = get_baseline_classes()
    high_temp = np.arange(395,405,0.1)
    ambient_temp = np.arange(293,303,0.1)
    conditions = {'high_temp':high_temp,'ambient_temp':ambient_temp,'low_rh':combo,'high_rh':combo2}
    build_class_categories(baseline,mode="Time Averaged",suffix="_test2_lowrh" + str(max(combo)) + "_highrh" + str(max(combo2)),conditions=conditions)

