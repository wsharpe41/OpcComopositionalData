import os

from Aerosol_Classes import aerosol_utils
from Aerosol_Classes.aerosol_utils import molar_to_molecular_mass
import math

import pandas as pd
from Aerosol_Classes.aerosolclass import AerosolClass
from Reverse_Simulator.aerosol_category import produce_category
import numpy as np
from Reverse_Simulator.classify_aerosol import map_test
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

def get_volatility_param(aerosol,t_ref,t_tdd):
    """
    Get volatility parameter of aerosol and td

    Args:
        t_tdd (int): Thermal denuder temp (K)
        t_ref (int): SVP reference temp (K)
        aerosol (AerosolClass): Aerosol of interest

    Returns:
        float: Volatility parameter
    """
    svp = aerosol.saturation_vapor_pressure[0]
    hvap = aerosol.hvap[0]
    m = aerosol.molar_mass[0]
    density = aerosol.density[0]
    r = 8.3145

    # Density must get bigger to compensate for MM change
    # SVP must get smaller to compensate for MM change
    vol = math.sqrt(m) * (svp/density) * (math.exp((-hvap*1000)/r*(1/t_tdd - 1/t_ref)))
    return vol

# Max volatility is 14995
# Min volatilitiy for volatile species is .00585
def get_usable_volatility_range():
    """
    Get usable volatility range for experiments

    Returns:
        None
    """
    vol_test = AerosolClass(name="Vol_test", gamma=1., saturation_vapor_pressure=[.01], molar_mass=[150], hvap=[100],
                            particle_diameter=[1.], kappa=[0], reference_temp=[400], density=[1.2], gsd=[1.4])
    final_diam = -99.
    init_diam = 1.
    while init_diam-final_diam > 0.1:
        t_svp = aerosol_utils.update_vapor_pressure(vol_test.saturation_vapor_pressure[0],initial_temp=400,final_temp=400,hvap=vol_test.hvap[0])

        d1 = (vol_test.mfr_at_temp(residence_time=10, time_step=.01, number_of_particles=1000, temperature=400,
                                   saturation_vapor_pressure=t_svp) * 10 ** 6)
        final_diam = d1
        print(f"Final Diameter: {final_diam}")
        vol = get_volatility_param(vol_test,400,440)
        print(f"Volatility : {vol}")
        vol_test.saturation_vapor_pressure[0] /= 1.01
        print(f"SVP: {vol_test.saturation_vapor_pressure[0]}")


def class_sensitivity(category, model=None):
    """
    Generate viz for sensitivity to each parameter

    category (str): Name of category
    model (keras.engine.training.Model): NN model to test accuracy

    Returns:
         None
    """
    if model is None:
        model = keras.models.load_model('neural_net')
    # Vary volatility
    cwd = os.path.dirname(os.path.abspath(os.getcwd()))
    directory_path = cwd + "/OpcSimResearch/Aerosol_Category_Outputs/slurm/" + category
    model_scores = {}
    history = []
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            data_point = pd.read_pickle(file_path)
            # Get conditions
            # File name after the first two underscores
            index = file_name.rfind("\\")
            shorter_file_name = file_name[index + 1:]
            split = shorter_file_name.split("_")
            conditions = '_'.join(split[2:])

            name = split[0] + "_" + split[1]
            test = data_point
            for df in test:
                df.drop(["rh","temp"],inplace=True,axis=1)
            test = np.asarray(test)
            y = map_test(test,name)
            predictions = (model.predict(test))
            most_likely = np.argmax(predictions, axis=1)
            test_acc = tf.keras.metrics.Accuracy()
            test_acc.update_state(y, most_likely)
            print(f" TEST ACCURACY: {test_acc.result()}")
            history.append(test_acc.result())
            model_scores[conditions] = test_acc.result()
    print(f"AVERAGE ACCURACY: {sum(history)/len(history)}")

    dnames = []
    dvalues = []
    knames = []
    kvalues = []
    gsdnames = []
    gsdvalues = []
    parnames = []
    parvalues = []
    volnames = []
    volvalues = []
    for key in model_scores.keys():
        if "diameter" in key:
            dnames.append(key[0:13])
            dvalues.append(model_scores[key])
        elif "hk" in key:
            knames.append(key[0:13])
            kvalues.append(model_scores[key])
        elif "gsd" in key:
            gsdnames.append(key[0:13])
            gsdvalues.append(model_scores[key])
        elif "vol" in key:
            split = key.split("_")
            print(key)
            #volname = '_'.join(split[2:])
            volname = split[0] + "_" + split[1]
            volnames.append(volname)
            volvalues.append(model_scores[key])
        elif "par" in key:
            parnames.append(key[0:15])
            parvalues.append(model_scores[key])
    plt.figure(1)
    plt.bar(range(len(dnames)), dvalues, tick_label=dnames)
    plt.xticks(rotation=30, ha='right')
    plt.figure(2)
    plt.bar(range(len(knames)), kvalues, tick_label=knames)
    plt.xticks(rotation=30, ha='right')
    plt.figure(3)
    plt.bar(range(len(gsdnames)), gsdvalues, tick_label=gsdnames)
    plt.xticks(rotation=30, ha='right')
    plt.figure(4)
    plt.title("Prediction Accuracy vs Volatility")
    plt.ylabel("Prediction Accuracy")
    plt.bar(range(len(volnames)), volvalues, tick_label=volnames)
    plt.xticks(rotation=30, ha='right')
    plt.figure(5)
    plt.bar(range(len(parnames)), parvalues, tick_label=parnames)
    plt.xticks(rotation=30, ha='right')
    plt.show()
    return

def build_class_categories():
    """
    Generate a small amount of example data

    Returns:
        None
    """

    mode = 'Median_Diameter'
    produce_category("NvHk_VHk_modes_4exp_p2", hvap=[[0.], [225.]], svp=[[0.], [200.]], kappa=[[1.], [1.]],
                     mode=mode, num_par=[1000, 1000], mm=[[130.], [130.]], svp_ref=[405, 405],
                     density=[[1.8], [1.8]], diameter_range=[[np.arange(.5, 10., .1)], [np.arange(.5, 10., .1)]])
    print("-----------------------------NvHk_VHk Created-----------------------------")
    produce_category("NvHk_NvLk_modes_4exp_p2", hvap=[[0.], [0.]], svp=[[0.], [0.]], kappa=[[1.], [.002]],
                     mode=mode, num_par=[1000, 1000], mm=[[130.], [130.]], svp_ref=[405, 405],
                     density=[[1.8], [1.8]], diameter_range=[[np.arange(.5, 10., .1)], [np.arange(.5, 10., .1)]])
    print("-----------------------------NvHk_NvLk Created-----------------------------")
    produce_category("NvHk_VLk_modes_4exp_p2", hvap=[[0.], [225.]], svp=[[0.], [200.]], kappa=[[1.], [.002]],
                     mode=mode, num_par=[1000, 1000], mm=[[130.], [130.]], svp_ref=[405, 405],
                     density=[[1.8], [1.8]], diameter_range=[[np.arange(.5, 10., .1)], [np.arange(.5, 10., .1)]])
    print("-----------------------------NvHk_VLk Created-----------------------------")
    produce_category("VHk_VLk_modes_4exp_p2", hvap=[[225.], [225.]], svp=[[200.], [200.]], kappa=[[1.], [.002]],
                     mode=mode, num_par=[1000, 1000], mm=[[130.], [130.]], svp_ref=[405, 405],
                     density=[[1.8], [1.8]], diameter_range=[[np.arange(.5, 10., .1)], [np.arange(.5, 10., .1)]])
    print("-----------------------------VHk_VLk Created-----------------------------")
    produce_category("VHk_NvLk_modes_4exp_p2", hvap=[[225.], [0.]], svp=[[200.], [0.]], kappa=[[1.], [.002]],
                     mode=mode, num_par=[1000, 1000], mm=[[130.], [130.]], svp_ref=[405, 405],
                     density=[[1.8], [1.8]], diameter_range=[[np.arange(.5, 10., .1)], [np.arange(.5, 10., .1)]])
    print("-----------------------------VHk_NvLk Created-----------------------------")
    produce_category("VLk_NvLk_modes_4exp_p2", hvap=[[225.], [0.]], svp=[[200.], [0.]], kappa=[[.002], [.002]],
                     mode=mode, num_par=[1000, 1000], mm=[[130.], [130.]], svp_ref=[405, 405],
                     density=[[1.8], [1.8]], diameter_range=[[np.arange(.5, 10., .1)], [np.arange(.5, 10., .1)]])