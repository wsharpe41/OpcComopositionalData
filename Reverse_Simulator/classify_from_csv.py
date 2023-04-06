from Experimental_Data import read_in_experimental_data
from Forward_Simulator import construct_realistic_output
import pandas as pd
import os
import quantaq
from quantaq.utils import to_dataframe

def format_file(path, mode,bins=24):
    """
    Format modulair csvs for use in classification

    Args:
        path (str): Path to file
        mode (str): Data processing mode
        bins (int): Number of bins in OPC
    Returns:
        pandas.Dataframe: Containing
    """
    files = os.listdir(path)
    # 4 excel sheets in directory
    full_output = []
    initial_size = -99.
    for file in files:
        exp_data = read_in_experimental_data.read_in_exp(path + "/" + file)
        # Cut out unnecessary columns
        # I want the bin columns
        # I want opc_temp + 273
        # I want opc_rh
        usable_data = pd.DataFrame()
        usable_data['rh'] = (exp_data['opc_rh'])
        usable_data['temp'] = (exp_data['opc_temp']) + 273
        for i in range(0,bins):
            usable_data['bin'+str(i)] = exp_data['bin' +str(i)]
        full_output.append(process_real_data(usable_data,mode,initial_size))
    return pd.DataFrame(full_output)

# It would have to be four modulairs not one four times
def get_from_modulairs(experiment_date,mode,devices, bins=24):
    """
    Format modulair csvs for use in classification

    Args:
        experiment_date (str): Date of experiment
        mode (str): Data processing mode
        bins (int): Number of bins in OPC
        devices (List[str]): Ids of modulairs
    Returns:
        List[pandas.Dataframe]: Containing
    """
    # Need to set up key as environment variable
    client = quantaq.QuantAQAPIClient(api_key="68KJ70S0CFJPUZVVGSKM9I6Y")
    initial_size = -99.
    full_output = []
    for device in devices:
        data = client.data.bydate(sn=device, date=experiment_date,raw=True)
        data = to_dataframe(data)
        usable_data = pd.DataFrame()
        usable_data['rh'] = (data['met.rh'])
        usable_data['temp'] = (data['met.temp']) + 273
        for i in range(0,bins):
            usable_data['bin'+str(i)] = data['opc.bin' +str(i)]
        usable_data = usable_data.fillna(0)
        # Pass into process real data with condtion for initial size
        real_data = process_real_data(usable_data,mode,initial_size)
        if type(real_data) is float:
            initial_size = real_data
        else:
            full_output.append(real_data)
    return full_output
    # If rh changes by more than 10% or if temp changes by more than 20k split data
    # Wait some minimum time 60 data points/5 minutes?, start searching again with new baseline


def process_real_data(usable_data,mode,initial_size):
    """
    Process data from OPC csvs

    Args:
        usable_data (List[pandas.Dataframe]): List of usable data from modulairs
        mode (str): Data processing mode
        initial_size (float): Initial particle size

    :return:
    """
    usable_data = construct_realistic_output.normalize_experiment_for_number_of_particles(usable_data)
    if mode == "Full Output":
        return usable_data
    usable_data = construct_realistic_output.average_simulated_experiments(usable_data)
    if mode == "Time Averaged":
        return usable_data
    rh = usable_data['rh']
    temp = usable_data['temp']
    average_size = construct_realistic_output.get_average_particle_size(usable_data)
    if mode == "Median Diameter":
        # Need Initial Size
        # Determine if file is baseline (room temp, room rh)
        # Need it to be in order of 400 k 20 rh, 290k 90rh, 400k 90rh,
        if initial_size == -99.:
            initial_size = average_size
            return initial_size
        else:
            size_change = (initial_size - average_size) / initial_size
            return {'average_size': average_size, 'rh': rh, 'temp': temp, 'size_change': size_change}