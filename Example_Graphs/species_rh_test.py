import opcsim
import matplotlib.pyplot as plt
import seaborn as sns


def get_rh_effects(aerosol):
    """
    Get effects of varying relative humidity on a given aerosol's size

    Args:
        aerosol (opcsim.Aerosol): Aerosol to graph

    Returns:
        None
    """
    # Alphasense R1
    opc = opcsim.OPC(wl=0.639, n_bins=24, dmin=0.35, dmax=40, theta=(32.0,88.0))
    opc.calibrate((1.5+0j), method="spline")
    data1 = opc.histogram(aerosol, weight="number", base="log10", rh=0.0)
    data2 = opc.histogram(aerosol, weight="number", base="log10", rh=40.0)
    data3 = opc.histogram(aerosol, weight="number", base="log10", rh=80.0)
    fig, ax = plt.subplots(3, figsize=(8, 8))

    ax[0] = opcsim.plots.histplot(data=data1, bins=opc.bins, ax=ax[0], label="RH=0%",
                               plot_kws=dict(linewidth=2, fill=True, alpha=.35, color='blue'))
    ax[1] = opcsim.plots.histplot(data=data2, bins=opc.bins, ax=ax[1], label="RH=40%",
                                  plot_kws=dict(linewidth=2, fill=True, alpha=.35, color='purple'))
    ax[2] = opcsim.plots.histplot(data=data3, bins=opc.bins, ax=ax[2], label="RH=80%",
                               plot_kws=dict(linewidth=2, fill=True, alpha=.35, color='orange'))

    ax[0].set_ylim(0, None)
    ax[0].set(xlabel="")
    ax[0].set_ylabel(fontsize=13,ylabel="dN/dlogDp (RH 0%)")
    ax[1].set(xlabel="")
    ax[1].set_ylabel(ylabel="dN/dlogDp (RH 40%)",fontsize=13)

    ax[2].set_xlabel(xlabel="Particle Diameter (um)", fontsize=13)
    ax[2].set_ylabel(fontsize=13, ylabel="dN/dlogDp (RH 80%)")

    sns.despine()
    plt.show()


def get_n3_bin_size():
    """
    Outputs particle diameters corresponding to AlphaSense N3 Bins

    Returns:
        numpy.ndarray: NpArray of each bin's min, max, and average particle diameters
    """
    opc = opcsim.OPC(wl=0.639, n_bins=24, dmin=0.35, dmax=40, theta=(32.0, 88.0))
    opc.calibrate((1.5 + 0j), method="spline")
    return opc.bins

def rh_effects_at_temp(aerosol,rh):
    """
    Returns effects of a given relative humidity on a given aerosol

    Args:
        aerosol (opcsim.Aerosol): Aerosol of interest
        rh (float): Relative humidity from 0-100

    Returns:
        pandas.Dataframe: OPC bin counts
    """
    # Alphasense R1
    opc = opcsim.OPC(wl=0.639, n_bins=24, dmin=0.35, dmax=40, theta=(32.0, 88.0))
    opc.calibrate((1.5 + 0j), method="spline")
    data = opc.histogram(aerosol, weight="number", base="log10", rh=rh)
    return data

def effects_of_experiment(volatile,diameters,kappas):
    """
    Plot the effects of passing an example aerosol through experimental array

    Args:
        volatile (List[bool]): List containing whether or not an example aerosol is volatile
        diameters (List[float]): List containing median diameters of example aerosols

    Returns:
        None
    """
    opc = opcsim.OPC(wl=0.639, n_bins=24, dmin=0.35, dmax=40, theta=(32.0, 88.0))
    opc.calibrate((1.5 + 0j), method="spline")
    aerosol = opcsim.AerosolDistribution("Test Dist")
    aerosol.add_mode(n=100, gm=diameters[0], gsd=1.5, label="One", refr=(1.5 + 0j), rho=1.8, kappa=kappas[0])
    aerosol.add_mode(n=10000, gm=diameters[1], gsd=1.5, label='Two', refr=(1.5 + 0j), rho=1.8, kappa=kappas[1])
    data1 = opc.histogram(aerosol, weight="number", base="log10", rh=20.0)
    data2 = opc.histogram(aerosol, weight="number", base="log10", rh=90.0)

    hot_aerosol = opcsim.AerosolDistribution("Test Dist")
    if volatile[0]:
        diameters[0] = 0.7*diameters[0]
    if volatile[1]:
        diameters[1] = 0.7 * diameters[1]
    hot_aerosol.add_mode(n=100, gm=diameters[0], gsd=1.5, label="One", refr=(1.5 + 0j), rho=1.8, kappa=kappas[0])
    hot_aerosol.add_mode(n=10000, gm=diameters[1], gsd=1.5, label='Two', refr=(1.5 + 0j), rho=1.8, kappa=kappas[1])
    data3 = opc.histogram(hot_aerosol, weight="number", base="log10", rh=20.0)
    data4 = opc.histogram(hot_aerosol, weight="number", base="log10", rh=90.0)

    hotter_aerosol = opcsim.AerosolDistribution("Test Dist")
    if volatile[0]:
        diameters[0] = 0.5 * diameters[0]
    if volatile[1]:
        diameters[1] = 0.5 * diameters[1]
    hotter_aerosol.add_mode(n=100, gm=diameters[0], gsd=1.5, label="One", refr=(1.5 + 0j), rho=1.8, kappa=kappas[0])
    hotter_aerosol.add_mode(n=10000, gm=diameters[1], gsd=1.5, label='Two', refr=(1.5 + 0j), rho=1.8, kappa=kappas[1])
    data5 = opc.histogram(hot_aerosol, weight="number", base="log10", rh=0.0)
    fig, ax = plt.subplots(5, figsize=(8, 8))

    ax[0] = opcsim.plots.histplot(data=data1, bins=opc.bins, ax=ax[0], label="RH=20%",
                                  plot_kws=dict(linewidth=2, fill=True, alpha=.35, color='blue'))
    ax[1] = opcsim.plots.histplot(data=data2, bins=opc.bins, ax=ax[1], label="RH=20%",
                                  plot_kws=dict(linewidth=2, fill=True, alpha=.35, color='purple'))
    ax[2] = opcsim.plots.histplot(data=data3, bins=opc.bins, ax=ax[2], label="RH=90%",
                                  plot_kws=dict(linewidth=2, fill=True, alpha=.35, color='orange'))
    ax[3] = opcsim.plots.histplot(data=data4, bins=opc.bins, ax=ax[3], label="RH=90%",
                                  plot_kws=dict(linewidth=2, fill=True, alpha=.35, color='orange'))
    ax[4] = opcsim.plots.histplot(data=data5, bins=opc.bins, ax=ax[4], label="RH=0%",
                                  plot_kws=dict(linewidth=2, fill=True, alpha=.35, color='red'))
    ax[0].set_ylim(0, None)
    ax[0].set(xlabel="")
    ax[0].set_ylabel(fontsize=13, ylabel="dN/dlogDp (RH 20%)")
    ax[1].set(xlabel="")
    ax[1].set_ylabel(ylabel="dN/dlogDp (RH 90%)", fontsize=13)
    ax[2].set_ylabel(fontsize=13, ylabel="dN/dlogDp (RH 90%)")

    ax[4].set_xlabel(xlabel="Particle Diameter (um)", fontsize=13)
    ax[4].set_ylabel(fontsize=13, ylabel="dN/dlogDp (RH 90%)")

    sns.despine()
    plt.show()