from Reverse_Simulator.produce_sensitivity_data import get_baseline_classes,build_class_categories
from multiprocessing.pool import Pool

def vary_particle_count():
    """
    Produces example data points with particle count varied from baseline conditions (1000 #/cc) and writes them to a pkl
    file

    Returns:
        None
    """
    pool = Pool(6)
    particle_counts = [1000,2000,3000,4000,5000,6000]
    pool.map(single_particle_count,particle_counts)

def single_particle_count(particle_count):
    """
    Takes in particle count and produces aerosol flows with that count

    Args:
        particle_count (int): Particle flow count (#/cc)
    Returns:
        None
    """
    # Vary uniform particle count (uniform sizes)
    # 1000, 3000, 6000
    baseline = get_baseline_classes()
    build_class_categories(aerosols=baseline,mode='Time Averaged',suffix="par___" + str(particle_count),num_par=[particle_count,particle_count,particle_count,particle_count])
    # 3 combos
    return

