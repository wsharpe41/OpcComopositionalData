import itertools
import math

from Aerosol_Classes.aerosolclass import AerosolClass
from Reverse_Simulator.aerosol_category import produce_category
import numpy as np
from itertools import combinations_with_replacement, permutations


def get_baseline_classes():
    """
    Get four property derived categories

    Returns:
        List[AerosolClass]: Four property derived cats
    """

    gsd_range = np.arange(1.2, 1.6, .01)
    particle_diameter_range = np.arange(.5, 10., .1)

    nv_lk = AerosolClass(name="NvLk", gamma=1, saturation_vapor_pressure=[0.], molar_mass=[130.], hvap=[225],
                         particle_diameter=[particle_diameter_range],
                         kappa=[.01], reference_temp=400, density=[1.8], gsd=[gsd_range])
    nv_hk = AerosolClass(name="NvHk", gamma=1, saturation_vapor_pressure=[0.], molar_mass=[130.], hvap=[225],
                         particle_diameter=[particle_diameter_range],
                         kappa=[1.], reference_temp=400, density=[1.8], gsd=[gsd_range])
    v_lk = AerosolClass(name="VLk", gamma=1, saturation_vapor_pressure=[200.], molar_mass=[130.], hvap=[225],
                        particle_diameter=[particle_diameter_range],
                        kappa=[.01], reference_temp=400, density=[1.8], gsd=[gsd_range])
    v_hk = AerosolClass(name="VHk", gamma=1, saturation_vapor_pressure=[200.], molar_mass=[130.], hvap=[225],
                        particle_diameter=[particle_diameter_range],
                        kappa=[1.], reference_temp=400, density=[1.8], gsd=[gsd_range])
    return [nv_lk, nv_hk, v_lk, v_hk]

def get_actual_classes():
    """
    Get four compound derived categories

    Returns:
        List[AerosolClass]: Four compound derived cats
    """
    gsd_range = np.arange(1.2, 1.6, .01)
    illite = AerosolClass(name="Dust",hvap=[178], saturation_vapor_pressure=[0.], kappa=[0.002],
                          molar_mass=[389.34], reference_temp=293,density=[2.75],particle_diameter=[np.arange(1, 10, .25)],
                        gsd=[gsd_range], gamma=1,num_par=500)
    smog = AerosolClass(name="Smog", hvap=[130.], saturation_vapor_pressure=[3.75E-10], kappa=[.578],
                        molar_mass=[132.14], reference_temp=298,
                        density=[1.2745],
                        particle_diameter=[np.arange(.25, .5, .01)],
                        gsd=[gsd_range], gamma=1,num_par=10000)
    sea_salt = AerosolClass(name="Sea_Salt", hvap=[255.], saturation_vapor_pressure=[.615], kappa=[1.06],
                        molar_mass=[133.322], reference_temp=1100,
                        density=[2.16],
                        particle_diameter=[np.arange(1, 2.5, .1)],
                        gsd=[gsd_range], gamma=1, num_par=1000)
    biomass_burning = AerosolClass(name="Biomass_Burning", hvap=[92.3], saturation_vapor_pressure=[.142], kappa=[.165],
                        molar_mass=[162.14], reference_temp=368,
                        density=[1.69],
                        particle_diameter=[np.arange(.25, 1, .01)],
                        gsd=[gsd_range], gamma=1, num_par=5000)

    return [illite,smog,sea_salt,biomass_burning]

def make_params_from_volatility(volatility):
    """
    Gets SVP from volatility parameter

    Args:
        volatility (float): Volatility parameter value

    Returns:
        float: saturation vapor pressure in Pa
    """
    # Molar mass, svp, density, hvap are varied and ttd are tref are set
    density = 1.8
    hvap = 225
    molar_mass = 130.
    r = 8.3145
    t_ref = 400
    ttd = 400
    # Just vary svp
    svp = (volatility * density / math.sqrt(molar_mass)) * 1 / (math.exp((-hvap * 1000) / r * (1 / ttd - 1 / t_ref)))
    return svp


def vary_differing_particle_diameter():
    """
    Produce data points with median diameters from baseline

    Returns:
        None
    """
    # Vary relative differences
    # Give each category a different one of the smaller four ranges above
    baseline = get_baseline_classes()
    particle_diameter_ranges = [np.arange(.25, 1., .01), np.arange(.5, 2., .01), np.arange(1., 4., .01),
                                np.arange(2., 8., .1)]
    i = 0
    for combo in permutations(particle_diameter_ranges, 4):
        diam_0 = combo[0]
        diam_1 = combo[1]
        diam_2 = combo[2]
        diam_3 = combo[3]
        par_0 = (diam_1.mean() / diam_0.mean()) ** 3 * 1000
        par_1 = 1000
        par_2 = (diam_1.mean() / diam_2.mean()) ** 3 * 1000
        par_3 = (diam_1.mean() / diam_3.mean()) ** 3 * 1000
        baseline[0].particle_diameter = [diam_0]
        baseline[1].particle_diameter = [diam_1]
        baseline[2].particle_diameter = [diam_2]
        baseline[3].particle_diameter = [diam_3]
        print([par_0, par_1, par_2, par_3])
        build_class_categories(baseline, "diff_diameter_" + str(i), num_par=[par_0, par_1, par_2, par_3],
                               mode='Multi-Component')
        i += 1
        # Adjust baseline aerosol
        # Whichever got category 2 remains as 1000, 1 is multiplied by 4, 3 is divided by 4, 4 is divided 16

    return


def produce_aerosol_category(aerosols, mode, name, num_par=None,conditions=None):
    """
    Produce aerosols under given conditions and mode

    Args:
    aerosols (List[AerosolClass]): List of aerosols to produce
    mode (str): Data processing mode
    num_par (int): Particles per cubic centimeter
    conditions (Dict[str,float]): Experimental conditions

    Returns:
        None
    """
    if aerosols[0].num_par is None or aerosols[1].num_par is None:
        num_par = [1000, 1000]
    else:
        num_par = [aerosols[0].num_par,aerosols[1].num_par]

    # Vary temps
    # Vary rhs
    if conditions is not None:
        # Low,low,high,high,low
        rh_range = [20,20,
                    90,90,0]
        temp_range = [298,350,
                      298,350,400]
        print(rh_range)
        print(temp_range)
        produce_category(name=name, hvap=[aerosols[0].hvap, aerosols[1].hvap],
                         svp=[aerosols[0].saturation_vapor_pressure, aerosols[1].saturation_vapor_pressure],
                         kappa=[aerosols[0].kappa, aerosols[1].kappa], mode=mode, num_par=num_par,
                         mm=[aerosols[0].molar_mass, aerosols[1].molar_mass],
                         svp_ref=[aerosols[0].reference_temp, aerosols[1].reference_temp],
                         density=[aerosols[0].density, aerosols[1].density],
                         diameter_range=[aerosols[0].particle_diameter, aerosols[1].particle_diameter],
                         gsd=[aerosols[0].gsd, aerosols[1].gsd],rh_range=rh_range,temp_range=temp_range,aerosols=aerosols)
    # Write to db
    else:
        produce_category(name=name, hvap=[aerosols[0].hvap, aerosols[1].hvap],
                         svp=[aerosols[0].saturation_vapor_pressure, aerosols[1].saturation_vapor_pressure],
                         kappa=[aerosols[0].kappa, aerosols[1].kappa], mode=mode, num_par=num_par,
                         mm=[aerosols[0].molar_mass, aerosols[1].molar_mass],
                         svp_ref=[aerosols[0].reference_temp, aerosols[1].reference_temp],
                         density=[aerosols[0].density, aerosols[1].density],
                         diameter_range=[aerosols[0].particle_diameter, aerosols[1].particle_diameter],
                         gsd=[aerosols[0].gsd, aerosols[1].gsd],aerosols=aerosols)
    

def build_baseline_categories():
    """
    Produce compound derived category examples

    Returns:
        None
    """
    aerosols = get_baseline_classes()
    # Want this to just get passed in n aerosols and it does every combination of them
    for combo in combinations_with_replacement(aerosols, 2):
        name = combo[0].name + "_" + combo[1].name
        produce_aerosol_category(combo, mode="Time Averaged", name=name)
    return


def build_class_categories(aerosols, suffix, mode, num_par=None,conditions=None):
    """
    Build property derived categories and write to pkl file

    aerosols (List[AerosolClass]): List of aerosols to produce
    suffix (str): Suffix for pkl output file
    mode (str): Data processing mode
    num_par (int): Particles per cubic centimeter
    conditions (Dict[str,float]): Experimental conditions
    Returns:
        None
    """
    # Want this to just get passed in n aerosols and it does every combination of them
    for combo in combinations_with_replacement(aerosols, 2):
        name = combo[0].name + "_" + combo[1].name + "_" + suffix
        if num_par is None:
            produce_aerosol_category(combo, mode=mode, name=name, num_par=num_par,conditions=conditions)

        else:
            particle_count = [num_par[aerosols.index(combo[0])], num_par[aerosols.index(combo[1])]]
            produce_aerosol_category(combo, mode=mode, name=name, num_par=particle_count,conditions=conditions)
        print("---------CATEGORY COMPLETE-----------")
