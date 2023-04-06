# Assumptions
import math
import numpy as np
import matplotlib.pyplot as plt
gamma_c = 1.0
# Need molecular mass

# Need Boltzmann's constant
# Need Temperature (changing)

# Just use carbon 13 as an example Saturation pressure at 25C = 10^-5 Pa

# Assume residence time of ten seconds

# Do I just want E?
# E  = gamma*sat_vapor_pres = math.sqrt(2*math.pi*molecular_mass*bc*temp) * SA

# Temp in Kelvin, gamma (0,1), sat_vapor_pres at given temp, molar mass in g/mol
def HertzKnudsen(gamma,sat_vapor_pres,molar_mass,temp):
    bc = 1.38 * 10 ** -23
    molecular_mass = (molar_mass/1000)/(6.022*10**23)
    jc = (gamma * sat_vapor_pres) / math.sqrt(2*math.pi*molecular_mass*bc*temp)
    return jc

# H Vap is joules, T in Kelvin
def extrapolateVaporPressure(ti,tf,vapor_press, hvap):
    r = 8.3145
    return vapor_press * math.exp(((-hvap*1000)/r)*((1/tf) - (1/ti)))

# Find flux over temperature range
def flux_over_range(t_start,t_finish, t_step, gamma, h_vap_lookup,molar_mass, t_boil,reference_temp,sat_vapor_pressure):
    flux_values = []
    temperatures = []
    current_temp = t_start
    # To get E(T) just multiply by surface area?
    while current_temp < t_finish:
        if current_temp > t_boil:
            current_vapor_pressure = extrapolateVaporPressure(reference_temp,current_temp,sat_vapor_pressure,h_vap_lookup)
            print(current_vapor_pressure)
            flux_values.append(HertzKnudsen(gamma,current_vapor_pressure,molar_mass,current_temp))
        else:
            flux_values.append(0)
        temperatures.append(current_temp)
        current_temp += t_step

    plt.plot(temperatures,flux_values)
    plt.ylabel("Evaporation Flux in molecules per m^2s")
    plt.xlabel("TD temp")
    plt.show()
    return np.column_stack((temperatures,flux_values))


# Initial mass in grams, residence time in seconds, particle diameter in m, flux in molecule/m^2s, t_step and residence_time in s, density in kg/m^3
def mfr_at_temp(saturation_pressure,particle_diameter,temperature,molar_mass, t_step,residence_time,density):
    flux = calculate_flux(saturation_pressure,temperature,molar_mass)
    sa = calculate_initial_surface_area(particle_diameter/2)
    t = 0
    mass = calculate_mass(density,sa)
    while t < residence_time:
        mass-=calculate_mass_loss(flux,t_step,sa,mass)
        t+=t_step
        sa = calculate_surface_area(mass,density)

    return mass

# Takes molar mass (g/mol) and returns molecular mass (kg/molecule)
def molar_to_molecular_mass(molar):
    return (molar / 6.022E23)/1000

# Takes saturation pressure (Pa), Temperature (K), molecular mass (kg/molecule) and returns Flux (molecules m^-2 s^-1)
def calculate_flux(saturation_pressure,temp,molecular_mass,gamma=1):
    boltzmann_cons = 1.38E-23
    return (gamma*saturation_pressure)/math.sqrt(2*math.pi*molecular_mass*boltzmann_cons*temp)

# Inputs particle size (radius) returns surface area assuming sphere
def calculate_initial_surface_area(particle_size):
    return particle_size**2 * 4 * math.pi

# Calculates the mass loss (kg) over a defined time step at a defined flux
def calculate_mass_loss(flux,t_step,surface_area,mass_initial):
    evaporation_rate = surface_area * flux
    return mass_initial - evaporation_rate * t_step

# Calculates the surface area after mass loss, assuming spherical
def calculate_surface_area(mass,density):
    volume = mass/density
    radius = ((3*volume)/(4*math.pi)) ** (1/3)
    return 4*math.pi*radius**2

def calculate_mass(density,surface_area):
    radius = math.sqrt(surface_area/(4*math.pi))
    volume = 4/3 * math.pi*radius**3
    return density * volume