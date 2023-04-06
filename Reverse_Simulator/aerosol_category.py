import numpy as np
import random
import pickle
import pandas as pd
import os
from typing_extensions import (
    Literal,
)
# Create class for an aerosol category
# Build all of the classes I want
# Nonvolatile low hygroscopicity, Volatile low hygroscopicity, volatile high hygroscopicity, nonvolatile high hygroscopicity
# Will have to create categories based on where the pollution came from
# Generate a bunch of different examples of each class
from Aerosol_Classes.aerosolclass import AerosolClass
from Forward_Simulator import construct_realistic_output
from Forward_Simulator.construct_realistic_output import simulate_under_given_conditions
from Forward_Simulator.database import populate_db, populate_postgres_db

def get_category_outputs(category_name):
    """
    Method to return outputs of a given category by name

    Args:
        category_name (str): Name of category of interest

    Returns:
        pandas.Dataframe: DataFrame containing example output of given category
    """
    return pd.read_pickle("Aerosol_Category_Outputs/" + category_name + "/" + category_name + "_category_examples.pkl")


class AerosolCategory:
    """Class for the generation of example aerosols within a category (e.g. Dust, Biomass Burning)"""

    def __init__(self, svp, mm_range, hvap_range, svp_ref, diameter_range, kappa_range, density_range,
                 geometric_standard_deviation_range, temp_range, rh_range, gamma=1, residence_time=10,
                 sampling_rate=5, time_step=.01, number_of_particles=None):
        """

        :param List(float) svp: Saturation Vapor Pressure of Aerosol in category in Pa
        :param List(float) mm_range: Range of molar masses for this category in g/mol
        :param List(float) hvap_range: Range of enthalpy of vaporization for this category in kj/mol
        :param svp_ref: Temperature for which the svp was found in K
        :param diameter_range: Range of median diameters of the particles in category in um
        :param kappa_range: Range of kappa values (hygroscopicity) for this category (no units)
        :param density_range: Range of possible densities for this category in g/cm^3
        :param geometric_standard_deviation_range: Range of standard deviations of aerosol distribution (no units)
        :param temp_range: Range of temperatures for these experiments in K
        :param rh_range: Range of relative humidities for these experiments (no units)
        :param gamma: Assumed 1 at all times
        :param residence_time: Residence time of thermal denuder in seconds
        :param sampling_rate: Frequency of OPC sampling in seconds, Alphasense N3 = 5
        :param time_step: Time step for thermal denuder calculations in s (see aerosolclass for more info)
        :param number_of_particles: Number of particles in flow in #/cc, deprecated
    """
        if number_of_particles is None:
            number_of_particles = [1000]
        self.svp = svp
        self.mm_range = mm_range
        self.hvap_range = hvap_range
        self.svp_ref = svp_ref
        self.diameter_range = diameter_range
        self.kappa_range = kappa_range
        self.density_range = density_range
        self.gsd = geometric_standard_deviation_range
        self.gamma = gamma
        self.rt = residence_time
        self.sampling_rate = sampling_rate
        self.ts = time_step
        self.num_par = number_of_particles
        self.temp_range = temp_range
        self.rh_range = rh_range

    def produce_category_outputs(self, number_of_outputs, category_name, experimental_length,
                                 mode=Literal['Full Output', 'Time Averaged', 'Median Diameter'],aerosols=None):
        """
        This method takes in an aerosol category, it's conditions and information on the experiments being performed
        and outputs the results of that experiment in one of three ways to a specified file

        :param (int) number_of_outputs: The amount of example experimental outputs you want generated
        :param (str) category_name:  The name of the aerosol category (e.g. "Dust", "Biomass_Burning")
        :param (int) experimental_length: The length of each experiment in seconds
        :param (int) experiments_per_output: The length of the experiment in seconds
        :param (literal) mode: Full Output mode should be used to get full times series bin counts, this provides the most data can be unruly to work with
                            Time Averaged mode should be used to get the average bin count for an experiment, output size will be number of bins + 2 (rh and temp)
                            Median Diameter mode should be used to get the average particle size over the course of an experiment as well as the change is size
                            relative to some initial conditions (by default temperature = 298K and relative humidity = 20%)
        :param bool aerosols: If this data should be written to pg db
        :return:
        """
        """
        
        Args:
            number_of_outputs (int): The amount of example experimental outputs you want generated
            category_name (str): The name of the aerosol category (e.g. "Dust", "Biomass_Burning"
            experimental_length (int): The length of each experiment in seconds
            mode (literal): 
            experiments_per_output (int): The number of different conditions tested per aerosol datapoint

         Returns:
             null
         """

        # Assume mm, hvap, diameter, kappa, density, and gsd are lists
        # Experiment variables will be known (length, temp, rh)
        # Others will be held constant (number of particles, residence time, time_step,sampling_rate)
        category_outputs = []
        all_parameters = []
        # Hvap, molar_mass, kappa, density and svp should all have same length

        for i in range(0, number_of_outputs):
            print(i)
            # Need to choose a position in a list of list then get that position for each
            chosen_cat = random.randint(0, len(self.hvap_range) - 1)
            chosen_compound = random.randint(0, len(self.hvap_range[chosen_cat]) - 1)
            # These would all have to be lists of lists
            molar_mass = [lst[chosen_compound] for lst in self.mm_range]
            hvap = [lst[chosen_compound] for lst in self.hvap_range]
            # For each item in self.diameter range grab a random value
            diameter = []
            gsd = []
            for dr in self.diameter_range:
                diam = dr[0][random.randint(0, len(dr[0]) - 1)]
                if len(str(diam)) > 4:
                    diam = np.float64("{:.2f}".format(diam))
                diameter.append(diam)
            for g in self.gsd:
                deviation = g[0][random.randint(0, len(g[0]) - 1)]
                if len(str(deviation)) > 4:
                    deviation = np.float64("{:.2f}".format(deviation))
                gsd.append(deviation)
            # diameter = [list(random.choice(lst))[random.randint(0, len(lst[chosen_cat]) - 1)] for lst in self.diameter_range]
            kappa = [lst[chosen_compound] for lst in self.kappa_range]
            density = [lst[chosen_compound] for lst in self.density_range]
            # gsd = [list(random.choice(lst))[random.randint(0, len(lst[chosen_cat]) - 1)] for lst in self.gsd]
            svp = [lst[chosen_compound] for lst in self.svp]
            # print(diameter)
            # print(random.choice(self.diameter_range))
            if "Parameter Estimation" in mode:
                all_parameters.append(
                    {'molar_mass': molar_mass, 'hvap': hvap, 'kappa': kappa, 'density': density, 'svp': svp})
            # Make all of these inputs lists so that it works with multiple modes
            category_aerosol = AerosolClass(name=category_name + str(i), gamma=self.gamma,
                                            saturation_vapor_pressure=svp,
                                            molar_mass=molar_mass, hvap=hvap, particle_diameter=diameter, kappa=kappa,
                                            reference_temp=self.svp_ref, density=density, gsd=gsd, refractive_index=1.5)
            rh = self.rh_range
            temp = self.temp_range
            experimental_info = {'length': experimental_length, 'sr': self.sampling_rate}
            conditions = pd.DataFrame({'rh': rh, 'temp': temp})
            td_info = {'res': self.rt, 'ts': self.ts, 'np': self.num_par}
            if 'Median Diameter' in mode or 'Relative_Time_Averaged' in mode or 'Multi-Component' in mode:
                baseline_temp = [298]
                baseline_rh = [20.0]
                ic_conditions = pd.DataFrame({'rh': baseline_rh, 'temp': baseline_temp})

                if 'Multi-Component' in mode:
                    initial_size, initial_params = simulate_under_given_conditions(ic_conditions, category_aerosol,
                                                                                   experimental_info,
                                                                                   td_info, mode,
                                                                                   -99)
                    category_outputs.append(
                        construct_realistic_output.simulate_under_given_conditions(aerosol=category_aerosol,
                                                                                   experimental_info=experimental_info,
                                                                                   conditions=conditions,
                                                                                   td_info=td_info,
                                                                                   mode=mode,
                                                                                   initial_size=initial_size,
                                                                                   initial_params=initial_params
                                                                                   ))
                else:
                    initial_size = simulate_under_given_conditions(ic_conditions, category_aerosol,
                                                                   experimental_info,
                                                                   td_info, mode,
                                                                   -99)
            else:
                initial_size = 0
            if mode != 'Multi-Component':
                category_outputs.append(
                    construct_realistic_output.simulate_under_given_conditions(aerosol=category_aerosol,
                                                                               experimental_info=experimental_info,
                                                                               conditions=conditions,
                                                                               td_info=td_info,
                                                                               mode=mode,
                                                                               initial_size=initial_size
                                                                               ))
        folder = ""
        output = []
        if mode == 'Time Averaged':
            folder = "Average/Test/"
            output = category_outputs
            print(output)
        elif mode == "Full Output":
            folder = ""
            output = category_outputs
        elif mode == "Median Diameter":
            folder = "Median_Diameter/"
            output = category_outputs
        elif mode == "Parameter Estimation":
            folder = "Parameter_Estimation/"
            output = [category_outputs, all_parameters]
        elif mode == "Multi-Component":
            folder = "Multi-Component/"
            output = category_outputs
        elif mode == "Relative_Time_Averaged":
            folder = "Relative_Time_Averaged/"
            output = category_outputs

        if output and aerosols:
            # call populate_db here
            populate_db.populate_db(aerosols=aerosols,output=output,gsd=gsd,diameter=diameter)
            populate_postgres_db.populate_db(aerosols=aerosols,output=output,gsd=gsd,diameter=diameter)
        if output:
            if not os.path.exists("Aerosol_Category_Outputs/" + folder + category_name):
                os.makedirs("Aerosol_Category_Outputs/" + folder + category_name)
            with open(
                    "Aerosol_Category_Outputs/" + folder + category_name + "/" + category_name + "_category_examples.pkl",
                    "wb") as outfile:
                pickle.dump(output, outfile)
        return

def produce_category(name, hvap, num_par, svp, kappa, mode, mm=None, svp_ref=None, temp_range=None, rh_range=None,
                     density=None, diameter_range=None, gsd = None,aerosols=None):
    """
    Generic method for creation of an aerosol category

    Args:
        name (str): Name of Aerosol Category
        hvap (List[List[float]]): Enthalpies of Vaporization in kj/mol
        num_par (List[int]): Number of particles in flow
        svp (List[List[float]]): Saturation Vapor Pressures in Pa
        kappa (List[List[float]]): Kappa values for each compound
        mode (str): Name of data processing mode for aerosol category
        mm (List[List[float]]): Molar masses of possible compounds
        svp_ref (List[float]): List of compound reference values
        temp_range (List[float]): List of experiment temperatures in K
        rh_range (List[float]): List of experiment relative humidities from 0 to 100
        density (List[List[float]]): List of compound densities in g/mol
        diameter_range (List[List[float]]): List of possible diameters in um

    Returns:
        None
    """
    if svp_ref is None:
        svp_ref = [400, 400]
    if density is None:
        density = [[1.8], [1.8]]
    if mm is None:
        mm = [[100.], [100.]]
    if diameter_range is None:
        diameter_range = [[np.arange(.25, 1, .01)], [np.arange(2.5, 10, .1)]]
    if rh_range is None:
        rh_range = [20, 20, 90, 90, 0]
    if temp_range is None:
        temp_range = [298, 350, 298, 350, 400]
    if gsd is None:
        gsd = [[np.arange(1.2, 1.6, .01)],[np.arange(1.2, 1.6, .01)]]
    cat_aerosol = AerosolCategory(diameter_range=diameter_range,
                                  geometric_standard_deviation_range=gsd,
                                  temp_range=temp_range, rh_range=rh_range, kappa_range=kappa, svp=svp, svp_ref=svp_ref,
                                  mm_range=mm, hvap_range=hvap, density_range=density, number_of_particles=num_par)
    cat_aerosol.produce_category_outputs(number_of_outputs=3, category_name=name, experimental_length=600,
                                         mode=mode,aerosols=aerosols)


