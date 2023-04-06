import itertools

from Reverse_Simulator.produce_sensitivity_data import get_baseline_classes, build_class_categories
from multiprocessing.pool import Pool


def vary_kappa_classes():
    """
    Produces example data points with Kappa varied from baseline conditions (low=0,high=1) and writes them to a pkl
    file

    Returns:
        None
    """
    # Vary kappas (low and high)
    # 9 combos
    low_kappas = [.1,.2,.4]
    high_kappas = [.7,.9,1.1]
    combos =itertools.product(low_kappas, high_kappas)
    pool = Pool(9)
    pool.map(single_kappa_class,combos)
    return

def single_kappa_class(x):
    """
    Takes in low and high kappa values and produces data points

    Args:
        x List[float]: Kappa values

    Returns:
        None
    """
    baseline = get_baseline_classes()
    baseline[0].kappa = [x[0]]
    baseline[2].kappa = [x[0]]
    baseline[1].kappa = [x[1]]
    baseline[3].kappa = [x[1]]
    build_class_categories(baseline, suffix="_" + "hk_" + str(x[1]) + "_lk___" + str(x[0]), mode='Time Averaged')

