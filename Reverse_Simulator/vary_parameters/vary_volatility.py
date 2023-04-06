import time

from Reverse_Simulator.produce_sensitivity_data import get_baseline_classes, make_params_from_volatility,build_class_categories
from multiprocessing.pool import Pool

def vary_volatility_classes():
    """
    Produces example data points with high volatility varied from baseline conditions and writes them to a pkl file

    Returns:
        None
    """
    # Vary volatility
    # Nonvolatile doesn't need to be changed
    # Volatile: 10000, 1000, 100, 10, 1, .1
    volatilities = [.1,1,10,100,1000,10000]
    pool = Pool(6)
    start = time.time()
    pool.map(single_volatility_class, volatilities)
    print("Finished in: {:.2f}s".format(time.time()-start))
    return


def single_volatility_class(vol):
    """
    Takes in volatility values for volatile category and writes to pkl file

    Args:
        vol (int): Volatility values of high volatile example category

    Returns:
        None
    """
    baseline = get_baseline_classes()
    svp = make_params_from_volatility(vol)
    baseline[2].saturation_vapor_pressure = [svp]
    baseline[3].saturation_vapor_pressure = [svp]
    build_class_categories(baseline, suffix="vol___" + str(vol),
                           mode='Time Averaged')
    return

