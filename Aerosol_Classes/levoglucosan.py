from Aerosol_Classes import aerosolclass

def example_getLevoglucosan():
    """
    Example aerosol class for levoglucosan
    Returns:
        AerosolClass: Levoglucosan aerosol class
    """
    # Kappa from https://acp.copernicus.org/articles/7/1961/2007/acp-7-1961-2007.pdf
    # Refractive index is very much an estimate https://en.wikipedia.org/wiki/Levoglucosenone from levoglucosenone
    # Saturation vapor pressure is 0?
    # Hvap https://webbook.nist.gov/cgi/cbook.cgi?ID=C498077&Units=SI&Mask=4#Thermo-Phase
    return aerosolclass.AerosolClass(name="Levoglucasan", gamma=1, molar_mass=[162.14], kappa=[.165],
                                     saturation_vapor_pressure=[4.63e-5], reference_temp=[298], hvap=[92.2],
                                     particle_diameter=[1.0], density=[1.69], gsd=[1.5], refractive_index=(1.62 + 0j))
