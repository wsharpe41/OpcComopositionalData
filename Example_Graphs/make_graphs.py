# Baseline
def make_baseline_graph(aerosol,base_path,temp_min,temp_max,temp_step,residence_time,time_step,number_of_particles):
    """
    Make baseline graphs for presentation

    Args:
        aerosol (AerosolClass): Aerosol of interest
        base_path (str): Output path for graph
        temp_min (float): Minimum TD temperature in K
        temp_max (float): Maximum TD temperature in K
        temp_step (float): Temperature to loop through temp_min and temp_max in K
        residence_time (float): Residence time of TD in seconds
        time_step (float): Time step to discretely integrate results over in seconds
        number_of_particles (int) : #/CC of aerosol flow

    Returns:
        None
    """

    figure_path = base_path + "/Aerosol_Baselines/" + aerosol.name
    aerosol.create_mfr(temp_min=temp_min,temp_max=temp_max,temp_step=temp_step,residence_time=residence_time,time_step=time_step,number_of_particles=number_of_particles,figure_path=figure_path)

# Refractive Index
def make_refractive_index_graphs(aerosol,base_path,temp_min,temp_max,temp_step,residence_time,time_step,number_of_particles):
    """
    Make refractive index graphs for presentation

    Args:
        aerosol (AerosolClass): Aerosol of interest
        base_path (str): Output path for graph
        temp_min (float): Minimum TD temperature in K
        temp_max (float): Maximum TD temperature in K
        temp_step (float): Temperature to loop through temp_min and temp_max in K
        residence_time (float): Residence time of TD in seconds
        time_step (float): Time step to discretely integrate results over in seconds
        number_of_particles (int) : #/CC of aerosol flow

    Returns:
        None
    """
    figure_path = base_path + "/Refractive_index/low_ri_" + aerosol.name
    aerosol.refractive_index = 1.1
    aerosol.create_mfr(temp_min=temp_min, temp_max=temp_max, temp_step=temp_step, residence_time=residence_time,fig_num=50,
                       time_step=time_step, number_of_particles=number_of_particles, figure_path=figure_path)
    aerosol.refractive_index = 1.9
    figure_path = base_path + "/Refractive_index/high_ri_" + aerosol.name
    aerosol.create_mfr(temp_min=temp_min, temp_max=temp_max, temp_step=temp_step, residence_time=residence_time,
                       time_step=time_step, number_of_particles=number_of_particles, figure_path=figure_path, fig_num=500)


# Molar Mass
def make_molar_mass_graphs(aerosol,base_path,temp_min,temp_max,temp_step,residence_time,time_step,number_of_particles):
    """
    Make MM graphs for presentation

    Args:
        aerosol (AerosolClass): Aerosol of interest
        base_path (str): Output path for graph
        temp_min (float): Minimum TD temperature in K
        temp_max (float): Maximum TD temperature in K
        temp_step (float): Temperature to loop through temp_min and temp_max in K
        residence_time (float): Residence time of TD in seconds
        time_step (float): Time step to discretely integrate results over in seconds
        number_of_particles (int) : #/CC of aerosol flow

    Returns:
        None
    """
    figure_path = base_path + "/Molar_Mass/low_mm_" + aerosol.name
    truth_mm = aerosol.molar_mass
    aerosol.molar_mass = truth_mm/2
    print("MOLAR MASS " + str(aerosol.molar_mass))
    aerosol.create_mfr(temp_min=temp_min,temp_max=temp_max,temp_step=temp_step,residence_time=residence_time,time_step=time_step,number_of_particles=number_of_particles,figure_path=figure_path,fig_num=40)
    figure_path = base_path + "/Molar_Mass/high_mm_" + aerosol.name
    aerosol.molar_mass = truth_mm * 2
    aerosol.create_mfr(temp_min=temp_min,temp_max=temp_max,temp_step=temp_step,residence_time=residence_time,time_step=time_step,number_of_particles=number_of_particles,figure_path=figure_path, fig_num=400)

# HVap
def make_hvap_graphs(aerosol,base_path,temp_min,temp_max,temp_step,residence_time,time_step,number_of_particles):
    """
    Make HVAP graphs for presentation

    Args:
        aerosol (AerosolClass): Aerosol of interest
        base_path (str): Output path for graph
        temp_min (float): Minimum TD temperature in K
        temp_max (float): Maximum TD temperature in K
        temp_step (float): Temperature to loop through temp_min and temp_max in K
        residence_time (float): Residence time of TD in seconds
        time_step (float): Time step to discretely integrate results over in seconds
        number_of_particles (int) : #/CC of aerosol flow

    Returns:
        None
    """

    figure_path = base_path + "/HVap/low_hvap_" + aerosol.name
    truth_hvap = aerosol.hvap
    aerosol.hvap = truth_hvap/2
    aerosol.create_mfr(temp_min=temp_min,temp_max=temp_max,temp_step=temp_step,residence_time=residence_time,time_step=time_step,number_of_particles=number_of_particles,figure_path=figure_path,fig_num=10)
    figure_path = base_path + "/HVap/high_hvap_" + aerosol.name
    aerosol.hvap = truth_hvap * 2
    aerosol.create_mfr(temp_min=temp_min,temp_max=temp_max,temp_step=temp_step,residence_time=residence_time,time_step=time_step,number_of_particles=number_of_particles,figure_path=figure_path,fig_num=100)

# Saturation Vapor Pressure
def make_saturation_vapor_pressure_graphs(aerosol,base_path,temp_min,temp_max,temp_step,residence_time,time_step,number_of_particles):
    """
    Make SVP graphs for presentation

    Args:
        aerosol (AerosolClass): Aerosol of interest
        base_path (str): Output path for graph
        temp_min (float): Minimum TD temperature in K
        temp_max (float): Maximum TD temperature in K
        temp_step (float): Temperature to loop through temp_min and temp_max in K
        residence_time (float): Residence time of TD in seconds
        time_step (float): Time step to discretely integrate results over in seconds
        number_of_particles (int) : #/CC of aerosol flow

    Returns:
        None
    """
    figure_path = base_path + "/Saturation_Vapor_Pressure/low_svp_" + aerosol.name
    truth_svp = aerosol.saturation_vapor_pressure
    aerosol.saturation_vapor_pressure = truth_svp/2
    aerosol.create_mfr(temp_min=temp_min,temp_max=temp_max,temp_step=temp_step,residence_time=residence_time,time_step=time_step,number_of_particles=number_of_particles,figure_path=figure_path,fig_num=20)
    figure_path = base_path + "/Saturation_Vapor_Pressure/high_svp_" + aerosol.name
    aerosol.saturation_vapor_pressure = truth_svp * 2
    aerosol.create_mfr(temp_min=temp_min,temp_max=temp_max,temp_step=temp_step,residence_time=residence_time,time_step=time_step,number_of_particles=number_of_particles,figure_path=figure_path,fig_num=200)

# Density
def make_density_graphs(aerosol,base_path,temp_min,temp_max,temp_step,residence_time,time_step,number_of_particles):
    """
    Make density graphs for presentation

    Args:
        aerosol (AerosolClass): Aerosol of interest
        base_path (str): Output path for graph
        temp_min (float): Minimum TD temperature in K
        temp_max (float): Maximum TD temperature in K
        temp_step (float): Temperature to loop through temp_min and temp_max in K
        residence_time (float): Residence time of TD in seconds
        time_step (float): Time step to discretely integrate results over in seconds
        number_of_particles (int) : #/CC of aerosol flow

    Returns:
        None
    """
    figure_path = base_path + "/Density/low_den_" + aerosol.name
    truth_den = aerosol.density
    aerosol.density = truth_den/2
    aerosol.create_mfr(temp_min=temp_min,temp_max=temp_max,temp_step=temp_step,residence_time=residence_time,time_step=time_step,number_of_particles=number_of_particles,figure_path=figure_path,fig_num=30)
    figure_path = base_path + "/Density/high_den_" + aerosol.name
    aerosol.density = truth_den * 2
    aerosol.create_mfr(temp_min=temp_min,temp_max=temp_max,temp_step=temp_step,residence_time=residence_time,time_step=time_step,number_of_particles=number_of_particles,figure_path=figure_path,fig_num=300)

# Make all of them
def make_all_graphs(aerosol,base_path,temp_min,temp_max,temp_step,residence_time,time_step,number_of_particles):
    """
    Make all example graphs for presentation

    Args:
        aerosol (AerosolClass): Aerosol of interest
        base_path (str): Output path for graph
        temp_min (float): Minimum TD temperature in K
        temp_max (float): Maximum TD temperature in K
        temp_step (float): Temperature to loop through temp_min and temp_max in K
        residence_time (float): Residence time of TD in seconds
        time_step (float): Time step to discretely integrate results over in seconds
        number_of_particles (int) : #/CC of aerosol flow

    Returns:
        None
    """
    print("----------Making Saturation Vapor Pressure Graph----------")
    make_saturation_vapor_pressure_graphs(aerosol,base_path,temp_min,temp_max,temp_step,residence_time,time_step,number_of_particles)
    print("----------Making HVap Graph----------")
    make_hvap_graphs(aerosol,base_path,temp_min,temp_max,temp_step,residence_time,time_step,number_of_particles)
    print("----------Making Molar Mass Graph----------")
    make_molar_mass_graphs(aerosol,base_path,temp_min,temp_max,temp_step,residence_time,time_step,number_of_particles)
    print("----------Making Refractive Index Graph----------")
    make_refractive_index_graphs(aerosol,base_path,temp_min,temp_max,temp_step,residence_time,time_step,number_of_particles)
    print("----------Making Baseline Graph----------")
    make_baseline_graph(aerosol,base_path,temp_min,temp_max,temp_step,residence_time,time_step,number_of_particles)
    print("----------Making Density Graph----------")
    make_density_graphs(aerosol,base_path,temp_min,temp_max,temp_step,residence_time,time_step,number_of_particles)