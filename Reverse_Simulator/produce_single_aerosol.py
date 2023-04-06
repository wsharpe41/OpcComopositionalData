from itertools import combinations_with_replacement

from Reverse_Simulator.produce_sensitivity_data import produce_aerosol_category, get_actual_classes
from multiprocessing.pool import Pool

def build_all_categories():
    """
    Build all 10 two mode categories

    Returns:
         None
    """
    # Want this to just get passed in n aerosols and it does every combination of them
    aerosols = get_actual_classes()
    combos = list(combinations_with_replacement(aerosols,2))
    pool = Pool(10)
    names = [combo[0].name + "_" + combo[1].name+"_lower_rh" for combo in combos]
    pool.starmap(build_single_category,(zip(combos,names)))
    print("---------CATEGORY COMPLETE-----------")

def build_single_category(combo,name):
    """
    Build one two mode category

    Args:
        combo (List[Aerosols]):
        name (str):

    Returns:
         None
    """
    produce_aerosol_category(combo,mode='Time Averaged',name=name)

