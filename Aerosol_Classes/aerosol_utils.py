import math

# Calculates the mass loss (kg) over a defined time step at a defined flux
# Flux in molecule m^-2 s^-1, t_step in s, surface_area in m^3
def calculate_mass_loss(flux, t_step, surface_area,molecular_mass):
    """
    Calculates the mass loss (kg) over a defined time step at a defined flux

    Args:
        flux float): Particle flux in m^-2 s^-1
        t_step (float): Time step to discretely integrate volatility equations over in s
        surface_area (float): Particle surface area in m^3
        molecular_mass (float): Mass of a molecule in kg/molecule

    Returns:
        float: Molecular mass loss over time step in kg
    """
    molecule_loss = surface_area * flux * t_step
    return molecule_loss * molecular_mass

# Takes molar mass (g/mol) and returns molecular mass (kg/molecule)
def molar_to_molecular_mass(molar):
    """
    Takes molar mass (g/mol) and returns molecular mass (kg/molecule)

    Args:
        molar (float): Molar mass in g/mol
    Returns:

        float: Molecular mass in kg/molecule
    """
    return (molar / 6.022E23) / 1000

# Takes saturation pressure (Pa), Temperature (K), molecular mass (kg/molecule) and returns Flux (molecules m^-2 s^-1)
def calculate_flux(saturation_pressure, temp, molecular_mass, gamma=1.):
    """
    Calculates flux from svp, temp, and molecular mass
    Args:
        saturation_pressure (float): Saturation vapor pressure in Pa
        temp (float): TD Temperature in K
        molecular_mass (float): Molecular mass in kg/molecule
        gamma (float): Evaporation coefficient from 0-1

    Returns:
        float: Flux in molecules m^-2 s^-2
    """
    boltzmann_cons = 1.38E-23
    return (gamma * saturation_pressure) / math.sqrt(2 * math.pi * molecular_mass * boltzmann_cons * temp)

# Inputs particle size (diameter (um)) returns surface area in m^3 assuming sphere
def calculate_initial_surface_area(particle_size):
    """
    Calculates particle surface area from initial diameter assuming a spherical particle
    Args:
        particle_size (float): Particle diameter in um

    Returns:
        float: Particle Surface area in m^3
    """
    return (particle_size/(10**6)) ** 2 * math.pi

def final_diameter(mass, density,number_of_particles):
    """
    Calculates final particle diameter after mass loss
    Args:
        mass (float) : Mass of flow in kg
        density (float) : Aerosol density in g/cc
        number_of_particles (int): Total particles in flow #/cc

    Returns:
        float: Diameter of single molecule in m
    """
    # Convert density to kg/m^3
    mass/=number_of_particles
    density *= 1000
    volume = mass / density
    radius = ((3 * volume) / (4 * math.pi)) ** (1 / 3)
    return 2*radius

# Calculates the surface area after mass loss, assuming spherical mass in kg, density in g/cm^3
def calculate_surface_area(mass, density,number_of_particles):
    """
    Calculate total surface area of flow
    Args:
        mass (float) : Mass of flow in kg
        float density (float) : Aerosol density in g/cc
        number_of_particles (int): Total particles in flow #/cc

    Returns:
        float: Surface area of flow in m^3
    """
    mass/=number_of_particles
    density *= 1000
    volume = mass / density
    radius = ((3 * volume) / (4 * math.pi)) ** (1 / 3)
    return 4 * math.pi * radius ** 2 * number_of_particles

# Density in g/cm^3, surface_area in m^3
def calculate_mass(density, surface_area, number_of_particles):
    """
    Calculates mass of particle flow in kg
    Args:
        density (float): Aerosol density in g/cc
        surface_area (float): Surface area of a single particle in m^3
        number_of_particles (int): Total particles in flow #/cc

    Returns:
        float: mass of flow in kg
    """
    surface_area /=number_of_particles
    radius = math.sqrt(surface_area / (4 * math.pi))
    volume = 4 / 3 * math.pi * radius ** 3
    return 1000* density * volume * number_of_particles

# Clausius-Clapeyron Relation to get saturation vapor pressure at a given temp
# Pressure in Pa, temperatures in K, hvap in kj/mol
def update_vapor_pressure(initial_pressure,initial_temp,final_temp,hvap):
    """
    Gets vapor pressure at temperature from vapor pressure at other reference temperature
    Args:
        initial_pressure (float): Vapor pressure at reference temperature in Pa
        initial_temp (float): Reference temperature in K
        final_temp (float): Temperature of interest in K
        hvap (float): Enthalpy of vaporization in kj/mol

    Returns:
        float: Vapor Pressure in Pa
    """
    # Gas constant
    r = 8.3145
    hvap *= 1000
    return initial_pressure * math.exp((-hvap/r)*(1/final_temp - 1/initial_temp))

def diameters_to_true_bins(opc,diameters):
    """
    Returns bins in opc where mean particle diameter should reside
    Args:
        opc (opcsim.OPC): OPC object from opcsim that is sampling particles
        diameters List[List[float] : list of particle diameters

    Returns:
        List[int]: OPC bins that particle diameters should fall into
    """
    # 3 x N array
    bins = opc.bins
    true_bins = []
    all_true_bins = []
    for diameter in diameters:
        # Determine if diameter is in range, gonna be slow
        for d in diameter:
            i = 0
            diameter_found = False
            for row in bins:
                if row[0] < d < row[2]:
                    true_bins.append(i)
                    diameter_found = True
                    break
                i+=1
            if not diameter_found:
                true_bins.append(-1)
        all_true_bins.append(true_bins)
    return true_bins


def get_dry_diameter(rh,kappa,wet_diameter):
    """
    Returns dry diameter from wet diameter using k-kohler theory
    Args:
        rh (float): Flow Relative Humidity
        kappa (float): Kohler theory kappa value
        wet_diameter (float): Diameter of wet particle

    Returns:
        float: Dry Diameter
    :return Dry Diameter
    :rtype float
    """
    aw = rh / 100.
    return wet_diameter / math.pow(1 + kappa * (aw / (1 - aw)), 1./3.)