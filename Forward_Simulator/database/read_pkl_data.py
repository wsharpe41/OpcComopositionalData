import os

from Reverse_Simulator import produce_sensitivity_data
from Forward_Simulator.database import populate_postgres_db
import pandas as pd
import re
# Get Existing pkl files and then read it into the pg database

# Required inputs are aerosols, output, gsd, diameter

# Get baseline conditions

# Based on name grab two baseline aerosols relevant
# If a property is called out, change that property in the aerosols
# If the property is called out gsd and diameter change that not aerosol
# If it isn't, there is ambiguity in which gsd and diameter was chosen. Leave them null for now

def read_all_pkl_data(folder,actual=False):
    """
    Read all data from a folder containing pkl files and populate the postgres DB with that information
    :param str folder: Name of folder containing files
    :param bool actual: If these aerosols are compound derived (True) or property derived (False)
    :return:
    """
    cwd = os.path.dirname(os.path.abspath(os.getcwd()))
    directory_path = cwd + "/OpcSimResearch/Aerosol_Category_Outputs/slurm/" + folder
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            read_pkl_file(file_path,actual)
    return



def read_pkl_file(file,actual=False):
    """
    Read aerosols from pkl file and write it to db
    :param str file: Name of file
    :param bool actual: If these aerosols are compound derived (True) or property derived (False)
    :return:
    """
    output = pd.read_pickle(file)
    gsd = [1.2,1.2]
    diameter = [0.25,.25]
    if actual:
        aerosols = get_actual_aerosols(file)
        populate_postgres_db.populate_db(aerosols, output, gsd, diameter)
        return
    else:
        aerosols = get_example_aerosols(file)


    prop_name, prop_value = get_changed_property(file)
    if prop_name is not None:
        if prop_name == 'gsd':
            gsd = [prop_value, prop_value]
        elif prop_name == "diameter":
            diameter = [prop_value, prop_value]
        elif prop_name == 'kappa':
            for aerosol in aerosols:
                if "Lk" in aerosol.name:
                    aerosol.kappa = prop_value[1]
                else:
                    aerosol.kappa = prop_value[0]
        elif prop_name == 'vol':
            for aerosol in aerosols:
                if "Nv" not in aerosol.name:
                    aerosol.kappa = prop_value
        elif prop_name == 'par':
            for aerosol in aerosols:
                aerosol.num_par = prop_value
        # For rh
        elif prop_name == 'rh':
            for out in output:
                # Rows 0,1,4 are low
                # 2,3 are high
                out.loc[[0,1,4],'rh'] = prop_value[0]
                out.loc[[2,3],'rh'] = prop_value[1]
        # For temp
        elif prop_name == 'temp':
            for out in output:
                # Rows 0,2 are low
                # 1,3,4 are high
                out.loc[[0,2], 'temp'] = prop_value[0]
                out.loc[[1,3,4], 'temp'] = prop_value[1]
    return populate_postgres_db.populate_db(aerosols,output,gsd,diameter)

def get_changed_property(file_name):
    """
    For aerosols which have had properties changed from the baseline, find that property from the file name
    :param str file_name: Name of file
    :return:
        - name(str) :name of property changed
        - property_values List: changed property values
    """
    # Split file_name
    index = file_name.rfind("\\")
    float_pattern = r"\d+\.\d+"
    int_pattern = r'\d+'
    numbers = re.findall(float_pattern, file_name[index+1:])
    if '_gsd_' in file_name:
        if len(numbers)!=1:
            print("TOO MANY GSD NUMBERS FOUND IN FILE NAME")
            return None
        else:
            return "gsd", numbers[0]
    elif '_lowrh' in file_name and "_highrh" in file_name:
        if len(numbers)!=2:
            print("INCORRECT NUMBER OF RH VALUES IN FILE NAME")
            return None
        else:
            return 'rh', numbers
    elif '_vol_' in file_name:
        if len(numbers)!=1:
            numbers = re.findall(int_pattern, file_name[index + 1:])
            if len(numbers) != 1:
                print("TOO MANY VOLATILITY NUMBERS FOUND IN FILE NAME")
                return None
            return 'vol',numbers[0]
        else:
            return "vol", numbers[0]
    elif '_lowtemp' in file_name:
        if len(numbers) != 2:
            print("INCORRECT NUMBER OF TEMP VALUES IN FILE NAME")
            return None
        else:
            return 'temps', numbers
    elif '_hk_' in file_name:
        if len(numbers) != 2:
            print("INCORRECT NUMBER OF RH VALUES IN FILE NAME")
            return None
        else:
            return 'kappa', numbers
    elif '_par_' in file_name:
        numbers = re.findall(int_pattern, file_name[index + 1:])
        if len(numbers) != 1:
            print("INCORRECT NUMBER OF PARTICLE COUNTS IN FILE NAME")
            return None
        else:
            return "par", numbers[0]
    elif '_diameter_' in file_name:
        if len(numbers) != 1:
            print("INCORRECT NUMBER OF DIAMETERS IN FILE NAME")
            return None
        else:
            return "diameter", numbers[0]
    return None

def get_example_aerosols(file_name):
    """
    Get aerosols from file using the name of the pkl file (for bimodal flows)
    :param str file_name: Name of file
    :return: List of aerosols in file
    :rtype List[AerosolClass]
    """
    # Check
    baseline = produce_sensitivity_data.get_baseline_classes()
    aerosols = []
    index = file_name.rfind("\\")
    file_name = file_name[index+1:]
    if "NvHk" in file_name:
        if file_name.count("NvHk") == 2:
            return [baseline[1],baseline[1]]
        else:
            aerosols.append(baseline[1])
    if "NvLk" in file_name:
        if file_name.count("NvLk") == 2:
            return [baseline[0],baseline[0]]
        else:
            aerosols.append(baseline[0])
    if "VLk" in file_name:
        if file_name.count("VLk") == 2:
            return [baseline[2],baseline[2]]
        else:
            aerosols.append(baseline[2])
    if "VHk" in file_name:
        if file_name.count("VHk") == 2:
            return [baseline[3],baseline[3]]
        else:
            aerosols.append(baseline[3])
    return aerosols

def get_actual_aerosols(file_name):
    """
    Get aerosols from file using the name of the pkl file (for bimodal flows)

    Args:
        file_name (str): Name of file

    Returns:
        List[AerosolClass]: List of aerosols in file
    """

    baseline = produce_sensitivity_data.get_actual_classes()
    aerosols = []
    if "D" in file_name:
        if file_name.count("D") == 2:
            return [baseline[0], baseline[0]]
        else:
            aerosols.append(baseline[0])
    if "SS" in file_name:
        if file_name.count("SS") == 2:
            return [baseline[2], baseline[2]]
        else:
            aerosols.append(baseline[2])
    if "S_" or "_S" in file_name:
        if file_name.count("S_S") == 1 and file_name.count("S_SS") == 0:
            return [baseline[1], baseline[1]]
        else:
            # Check if that smog is ss
            if file_name.count("S") % 2 == 1:
                aerosols.append(baseline[1])
    if "BB" in file_name:
        if file_name.count("BB") == 2:
            return [baseline[3], baseline[3]]
        else:
            aerosols.append(baseline[3])
    return aerosols