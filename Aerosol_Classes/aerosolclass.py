import matplotlib.pyplot as plt
from Example_Graphs import species_rh_test
from Aerosol_Classes.aerosol_utils import *
from Forward_Simulator import construct_realistic_output
import opcsim
import pandas as pd
import os
import time
from random import random
import numpy as np


class AerosolClass:
    """Class used to represent an aerosol and a flow with all data needed to create an MFR and use OpcSim"""

    def __init__(self, name, gamma, saturation_vapor_pressure, molar_mass, hvap, particle_diameter,
                 kappa, reference_temp, density, gsd, refractive_index=(1.5 + 0j),num_par=None):
        """
        :param str name: Name of aerosol
        :param float gamma: Always assumed as 1
        :param List[float] saturation_vapor_pressure: Saturation vapor pressure of aerosol in Pa
        :param List[float] molar_mass: Molar Mass of aerosol in g/mol
        :param List[float] hvap: Enthalpy of Vaporization in kj/mol (does not matter if svap=0)
        :param List[float] particle_diameter: Median diameter of particle in flow in um
        :param List[float] kappa: Hygrospocicity measure without units
        :param List[float] reference_temp: Temperature at which the svp value was found in K
        :param List[float] density: Density of the aerosol in g/cm^3
        :param List[float] gsd: Geometric Standard Deviation of aerosol distribution, no units
        :param refractive_index: Complex refractive index of aerosol (eg. (1.5 + 0j)
        """
        # These will all have to be lists to accommodate multiple component aerosols
        self.name = name
        # No units
        self.gamma = gamma
        # Pascals
        self.saturation_vapor_pressure = saturation_vapor_pressure
        # grams/mol
        self.molar_mass = molar_mass
        # kj/mol
        self.hvap = hvap
        # um
        self.particle_diameter = particle_diameter
        # No units
        self.kappa = kappa
        # Kelvin, reference for SVP
        self.reference_temp = reference_temp
        # g/cm^3
        self.density = density
        self.gsd = gsd
        self.refractive_index = refractive_index
        self.num_par = num_par

    # Need to make this agnostic to number of particles somehow
    def mfr_at_temp(self, temperature, time_step, residence_time, number_of_particles, saturation_vapor_pressure,
                    index=0):
        """

        :param float temperature: Thermal denuder temperature in K
        :param float time_step: Time step to discretely integrate volatility equations over in s
        :param float residence_time: Residence time of thermal denuder in s
        :param float number_of_particles: Number of particles in flow #/cc
        :param float saturation_vapor_pressure: Saturation vapor pressure of aerosol in Pa
        :param int index: Index of mfr
        :return: median diameter of particle in flow after passing through thermal denuder in um
        :rtype float
        """

        # Flux remains the same no matter the particle density
        flux = calculate_flux(saturation_vapor_pressure, temperature, molar_to_molecular_mass(self.molar_mass[index]))
        # SA increases with nop
        sa = calculate_initial_surface_area(self.particle_diameter[index]) * number_of_particles
        t = 0
        # Initial mass increases with nop
        initial_mass = calculate_mass(self.density[index], sa, number_of_particles)
        mass = initial_mass
        while t < residence_time:
            # Depends on SA which depends on NOP
            mass -= calculate_mass_loss(flux, time_step, sa, molar_to_molecular_mass(self.molar_mass[index]))
            if mass < 0:
                return 0
            t += time_step
            # Depends on NOP
            sa = calculate_surface_area(mass, self.density[index], number_of_particles)
        # Output final radius instead
        return final_diameter(mass, self.density[index], number_of_particles)

    def create_mfr(self, temp_min, temp_max, temp_step, residence_time, time_step, number_of_particles,
                   figure_path="S:\PycharmProjects\OpcSimResearch\Figures/Aerosol_Baselines/", fig_num=1):
        """
        Output mass fraction remaining graph to a specified file path for a specified thermal denuder (TD) and aerosol across temperatures

        :param int temp_min: Minimum temperature of TD to be simulated in K
        :param int temp_max: Maximum temperature of TD to be simulated in K
        :param int temp_step: Step to move between min and max temperatures in experiment in K
        :param float residence_time: TD residence time in s
        :param float time_step: Time step to discretely integrate volatility equations over in s
        :param List[float] number_of_particles: Number of particles in flow #/cc
        :param str figure_path: Path to output MRF to
        :param int fig_num: MatPlotLib number of outputted figure
        :return:
        """
        # Loop this
        initial_vapor_pressure = self.saturation_vapor_pressure

        for j in range(0, 5):
            rh = float(j * 20)
            bin_dict = {}
            temperatures = []
            temp = temp_min
            print(rh)
            diameters = []
            while temp <= temp_max:
                aerosol_distribution = opcsim.AerosolDistribution("NaCl")
                temp_final_diameter = []
                for i in range(len(self.gsd)):
                    saturation_vapor_pressure = update_vapor_pressure(initial_vapor_pressure[i], self.reference_temp[i],
                                                                      temp,
                                                                      self.hvap[i])
                    temp_final_diameter.append(self.mfr_at_temp(temp, time_step, residence_time, number_of_particles[i],
                                                                saturation_vapor_pressure, index=i) * 10 ** 6)
                    print(temp_final_diameter)
                    if temp_final_diameter == 0:
                        continue
                    # This has to be a loop (outside of j)
                    aerosol_distribution.add_mode(n=number_of_particles[i], gm=temp_final_diameter[i], gsd=self.gsd[i],
                                                  label=self.name, refr=self.refractive_index, rho=self.density[i],
                                                  kappa=self.kappa[i])
                bin_assignments = species_rh_test.rh_effects_at_temp(aerosol_distribution, rh)
                # Create a a list for each bin
                print(temp)
                for i in range(0, bin_assignments.size):
                    bin_list = bin_assignments.tolist()
                    if i in bin_dict.keys():
                        bin_dict[i].append(bin_list[i])
                    else:
                        bin_dict[i] = [bin_list[i]]

                temperatures.append(temp)
                diameters.append(temp_final_diameter)
                temp += temp_step
            plt.rcParams["figure.figsize"] = (20, 10)
            plt.rcParams.update({'font.size': 16})

            plt.figure(j + fig_num)
            for key in bin_dict.keys():
                plt.plot(temperatures, bin_dict[key], label="Bin " + str(key), linestyle='--', marker='o')
            # This would break
            true_bins = diameters_to_true_bins(opcsim.OPC(wl=0.639, n_bins=16, dmin=0.35, dmax=12.4), diameters)
            plt.ylabel("dN/dlogDp", fontsize=20)
            plt.xlabel("Temperature (K)", fontsize=20)
            # Shrink current axis by 20%
            ax = plt.subplot(111)
            box = ax.get_position()
            ax.set_position([box.x0 + box.width * .25, box.y0, box.width, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(0, 0.5))
            ax2 = ax.twinx()
            # make a plot with different y-axis using second axis object
            ax2.plot(temperatures, true_bins, label="Truth", marker='x')
            ax2.set_ylabel("Correct Bin", loc="center", fontsize=20)
            ax2.set_position([box.width * .2, box.y0, box.width, box.height])

            # figure_path = 'S:\PycharmProjects\OpcSimResearch\Figures/Aerosol_Baselines/demo_'
            plt.savefig(figure_path + "_at_rh" + str(rh) + ".png")


    def simulate_constant_experiment(self, residence_time, time_step, number_of_particles, rh, temperature,
                                     experimental_length,
                                     sampling_rate, td_output_rh=0., ambient_rh=20., td_first=True, randomness=True):
        """
        Simulate passing an aerosol through a thermal denuder at a constant temperature for a given time

        :param float residence_time: TD residence time in s
        :param float time_step: Time step to discretely integrate volatility equations over in s
        :param List[float] number_of_particles: Number of particles in flow #/cc
        :param float rh: Relative humidity of flow in experiment from 0 to 100
        :param float temperature: TD temperature in K
        :param float experimental_length: Length of experiment in s
        :param float sampling_rate: Frequency of OPC sampling in s
        :param bool randomness: If randomness should be included in opc bin output
        :param float td_output_rh : Relative humidity of flow coming out of thermal denuder (assumed 0)
        :param bool td_first: Whether or not the thermal denuder is before the drier in the path of the flow
        :param float ambient_rh: RH of ambient flow before it enters the drier
        :return: DataFrame of OPC bin counts for each sample
        :rtype pandas.DataFrame
        """

        if not td_first:
            self.simulate_drier(self.particle_diameter, start_rh=ambient_rh, ending_rh=rh)
        bin_dict = {}
        all_zeros = True
        aerosol_distribution = opcsim.AerosolDistribution()

        for j in range(len(self.particle_diameter)):
            initial_vapor_pressure = self.saturation_vapor_pressure[j]
            saturation_vapor_pressure = update_vapor_pressure(initial_vapor_pressure, self.reference_temp[j],
                                                              temperature,
                                                              self.hvap[j])
            temp_final_diameter = self.mfr_at_temp(temperature, time_step, residence_time, number_of_particles[j],
                                                   saturation_vapor_pressure,index=j) * 10 ** 6
            # This would only be the case if td was before drier
            # This could be wrong
            if len(str(temp_final_diameter)) > 5:
                temp_final_diameter = np.float64("{:.3f}".format(temp_final_diameter))

            if td_first and td_output_rh != 0:
                temp_final_diameter = get_dry_diameter(td_output_rh, self.kappa[j], temp_final_diameter)
            if temp_final_diameter != 0:
                all_zeros = False

                aerosol_distribution.add_mode(n=number_of_particles[j], gm=temp_final_diameter, gsd=self.gsd[j],
                                              refr=self.refractive_index, rho=self.density[j], kappa=self.kappa[j])

        if all_zeros:
            bin_assignments = np.zeros(shape=24, dtype=float)
        else:
            bin_assignments = species_rh_test.rh_effects_at_temp(aerosol_distribution, rh)
        # Maybe better to introduce randomness here since it would be way faster, and have rh_effects_at_temp outside of loop
        # Create a a list for each bin at time t =0 to
        experimental_time = 0
        while experimental_time < experimental_length:
            for i in range(0, bin_assignments.size):
                bin_list = bin_assignments.tolist()
                # Just set randomness to 10% for no reason. Can tune in future
                if not randomness:
                    if "bin" + str(i) in bin_dict.keys():
                        bin_dict["bin" + str(i)].append(bin_list[i])
                    else:
                        bin_dict["bin" + str(i)] = [bin_list[i]]
                else:
                    if random() > 0.5:
                        scattered_value = 0.1 * random() * bin_list[i] + bin_list[i]
                    else:

                        scattered_value = bin_list[i] - 0.1 * random() * bin_list[i]
                    if "bin" + str(i) in bin_dict.keys():
                        bin_dict["bin" + str(i)].append(scattered_value)
                    else:
                        bin_dict["bin" + str(i)] = [scattered_value]
            experimental_time += sampling_rate
        bin_df = pd.DataFrame.from_dict(bin_dict)
        plt.show()
        bin_df['rh'] = rh
        bin_df['temp'] = temperature
        return bin_df

    # Should diameters be a list here?
    def simulate_drier(self, starting_diameters, start_rh, ending_rh):
        """
        Simulates the effects of a drier changing the rh of an incoming flow
        1. Takes median diameter as well as RH and then gets dry diameter
        2. Takes dry diameter and get simulated OPC output
        3. Estimate final diameter from that output
        :param List[float] starting_diameters: Diameter of particle before it enters the drier
        :param float start_rh: RH of ambient flow
        :param float ending_rh: RH of drier output
        :return:
        """
        for i in range(len(self.kappa)):
            dry_diameter = get_dry_diameter(start_rh, self.kappa[i], starting_diameters[i])
            # Pass dry diameter plus final rh into opcsim
            aerosol_distribution = opcsim.AerosolDistribution("Test_Distribution")
            # This has to be a loop
            aerosol_distribution.add_mode(n=1000, gm=dry_diameter, gsd=self.gsd[i], label=self.name,
                                          refr=self.refractive_index, rho=self.density[i], kappa=self.kappa[i])
            drier_opc_output = species_rh_test.rh_effects_at_temp(aerosol_distribution, ending_rh)
            final_drier_diameter = construct_realistic_output.get_average_particle_size(drier_opc_output)
            self.particle_diameter[i] = final_drier_diameter
        return
