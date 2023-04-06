# ML Classification Problem
# Take in a bunch of simulation generated graphs with a wide range of different parameters, as well as what class each belongs to
# Train a CNN or something else on those, aim for highest accuracy (Look at demos from class)
# Spit out prediction of class
import configparser
import sqlite3

import pandas as pd
import numpy as np
import psycopg2
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from supervised.automl import AutoML
import os
import pickle
import matplotlib.pyplot as plt
import glob


# Data is going to be split into 4 JSONs
# Going to start with averaged bin values
# Each experiment is going to have 4 data points in it with each datapoint having a 26? x 1 shape
# Inputs are bin values, rh, and temp
# Output is category

def make_classification_data():
    """Construct numpy arrays from data for four aerosol categories

    Returns:
        data (numpy.ndarray): Numpy array with all data from four categories
        classes (numpy.ndarray): Numpy array with classes of all data
    """
    cwd = os.path.dirname(os.path.abspath(os.getcwd()))
    # Read in data from each json as a dataframe
    bb = pd.read_pickle(
        cwd + "/OpcSimResearch/Aerosol_Category_Outputs/Median_Diameter/Biomass_Burning/Biomass_Burning_category_examples.pkl")
    bb = np.asarray(bb)
    dust = pd.read_pickle(
        cwd + "/OpcSimResearch/Aerosol_Category_Outputs/Median_Diameter/Dust/Dust_category_examples.pkl")
    dust = np.asarray(dust)
    ss = pd.read_pickle(
        cwd + "/OpcSimResearch/Aerosol_Category_Outputs/Median_Diameter/Sea_Salt/Sea_Salt_category_examples.pkl")
    ss = np.asarray(ss)
    smog = pd.read_pickle(
        cwd + "/OpcSimResearch/Aerosol_Category_Outputs/Median_Diameter/Urban_Smog/Urban_Smog_category_examples.pkl")
    smog = np.asarray(smog)
    print("SMOG")
    print(smog)
    print("Biomass Burning")
    print(bb)
    print("Dust")
    print(dust)
    print("Sea Salt Aerosol")
    print(ss)
    # Create Dataframe for all data
    classes = np.append(np.full(len(bb), 'Biomass_Burning'),
                        [np.full(len(dust), 'Dust'), np.full(len(ss), 'Sea_Salt'), np.full(len(smog), 'Urban_Smog')])

    data = np.concatenate((bb, dust, ss, smog))
    print(data.shape)
    return data, classes


def make_example_classification_data():
    """Construct numpy arrays from data for four example aerosol categories

    Returns:
        data (numpy.ndarray): Numpy array with all data from four categories
        classes (numpy.ndarray): Numpy array with classes of all data
    """
    cwd = os.path.dirname(os.path.abspath(os.getcwd()))
    # Read in data from each json as a dataframe
    bb = pd.read_pickle(
        cwd + "/OpcSimResearch/Aerosol_Category_Outputs/Median_Diameter/NvHk/NvHk_category_examples.pkl")
    bb = np.asarray(bb)
    dust = pd.read_pickle(
        cwd + "/OpcSimResearch/Aerosol_Category_Outputs/Median_Diameter/NvLk/NvLk_category_examples.pkl")
    dust = np.asarray(dust)
    ss = pd.read_pickle(
        cwd + "/OpcSimResearch/Aerosol_Category_Outputs/Median_Diameter/VLk/VLk_category_examples.pkl")
    ss = np.asarray(ss)
    smog = pd.read_pickle(
        cwd + "/OpcSimResearch/Aerosol_Category_Outputs/Median_Diameter/VHk/VHk_category_examples.pkl")
    smog = np.asarray(smog)
    # Create Dataframe for all data
    classes = np.append(np.full(len(bb), 'NvHk'),
                        [np.full(len(dust), 'NvLk'), np.full(len(ss), 'VLk'), np.full(len(smog), 'VHk')])
                         #np.full(len(mvhk), 'MvHk'), np.full(len(mvlk), 'MvLk'), np.full(len(mvmk), 'MvMk'),
                         #np.full(len(nvmk), 'NvMk'), np.full(len(vmk), 'VMk')])
    data = np.concatenate((bb, dust, ss, smog, #mvhk, mvlk, mvmk, nvmk, vmk
                        ))
    print(data)
    return data, classes


def make_time_average_classification_data():
    """Construct numpy arrays from data for four example aerosol categories

    Returns:
        data (numpy.ndarray): Numpy array with all data from four categories
        classes (numpy.ndarray): Numpy array with classes of all data
    """
    cwd = os.path.dirname(os.path.abspath(os.getcwd()))
    # Read in data from each json as a dataframe
    bb = pd.read_pickle(
        cwd + "/OpcSimResearch/Aerosol_Category_Outputs/Average/Nonvolatile_High_Kappa/Nonvolatile_High_Kappa_category_examples.pkl")
    bb = np.asarray(bb)
    dust = pd.read_pickle(
        cwd + "/OpcSimResearch/Aerosol_Category_Outputs/Average/Nonvolatile_Low_Kappa/Nonvolatile_Low_Kappa_category_examples.pkl")
    dust = np.asarray(dust)
    ss = pd.read_pickle(
        cwd + "/OpcSimResearch/Aerosol_Category_Outputs/Average/Volatile_Low_Kappa/Volatile_Low_Kappa_category_examples.pkl")
    ss = np.asarray(ss)
    smog = pd.read_pickle(
        cwd + "/OpcSimResearch/Aerosol_Category_Outputs/Average/Volatile_High_Kappa/Volatile_High_Kappa_category_examples.pkl")
    smog = np.asarray(smog)
    # Create Dataframe for all data
    classes = np.append(np.full(len(bb), 'NvHk'),
                        [np.full(len(dust), 'NvLk'), np.full(len(ss), 'VLk'), np.full(len(smog), 'VHk')])
    data = np.concatenate((bb, dust, ss, smog))
    return data, classes


def read_in_actual_multicomponent(folder, names, plot=False):
    """
    Generic method to read in and concatenate aerosol categories from pkl files

    Args:
        folder (str): Name of folder where pkl files are located
        names (List[str]): Name of files in folder

    Returns:
        data (numpy.ndarray): Numpy array with all data from four categories
        classes (numpy.ndarray): Numpy array with classes of all data
    """
    cwd = os.path.dirname(os.path.abspath(os.getcwd()))
    # Read in data from each json as a dataframe
    data = None
    classes = None
    for name in names:
        data_point = pd.read_pickle(
            cwd + "/OpcSimResearch/Aerosol_Category_Outputs/" + folder + "/" + name + "/" + name + "_category_examples.pkl")
        d_list = []
        print(len(data_point))
        for d in data_point:
            d_list.append(d)
        data_point = d_list
        # Plot this data to look for differences
        if plot:
            plt.figure(1)
            first_entries = [sublist['total_particles'][3] for sublist in data_point]
            plt.plot(first_entries)
            plt.figure(2)
            second_entries = [sublist['average_size'][0] for sublist in data_point]
            plt.plot(second_entries)

        if data is None:
            data = data_point
        else:
            data = np.concatenate((data, data_point))
        if classes is None:
            classes = np.full(len(data_point), name)
        else:
            classes = np.append(classes, np.full(len(data_point), name))
    if plot:
        plt.show()
    return data, classes


def concat_outputs(folder, path1, path2):
    """
    Concatenate two pkl output aerosol categories and write to first path

    Args:
        folder (str): Name of folder in which both files are located
        path1 (str): Name of first file
        path2 (str): Name of second file

    Returns:
        None
    """
    cwd = os.path.dirname(os.path.abspath(os.getcwd()))
    data1 = pd.read_pickle(
        cwd + "/OpcSimResearch/Aerosol_Category_Outputs/" + folder + "/" + path1 + "/" + path1 + "_category_examples.pkl")
    data2 = pd.read_pickle(
        cwd + "/OpcSimResearch/Aerosol_Category_Outputs/" + folder + "/" + path2 + "/" + path2 + "_category_examples.pkl")

    with open(
            "Aerosol_Category_Outputs/" + folder + "/" +path1 + "/" + path1 + "_category_examples.pkl",
            "wb") as outfile:
        pickle.dump(data1+data2, outfile)
    return


def classify_with_automl(data, classes):
    """Perform AutoMl classification with results output to folder

     Args:
        data (numpy.ndarray): Data to be classified
        classes (numpy.ndarray): Classes of data points in data

    Returns:
        supervised.automl.Automl: AutoML model fit to inputs
     """
    data = data.reshape(data.shape[0], -1)
    # scale data
    scaler = StandardScaler()
    scaled =scaler.fit_transform(data)
    automl = AutoML(mode="Explain")
    automl.fit(scaled, classes)
    return automl

def read_in_engaging_output(folder):
    """
    Generic method to read in and concatenate aerosol categories from pkl files

    Args:
        folder (str): Name of folder where pkl files are located

    Returns:
        X_train (numpy.ndarray): X training data for downstream ml
        X_test (numpy.ndarray): X testing data for downstream ml
        y_train (numpy.ndarray): y training data for downstream ml
        y_test (numpy.ndarray): y testing data for downstream ml
    """
    cwd = os.path.dirname(os.path.abspath(os.getcwd()))
    directory_path = cwd + "/OpcSimResearch/Aerosol_Category_Outputs/Slurm/" + folder
    # Read in data from each json as a dataframe
    data = None
    classes = None
    test_data = {}
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            data_point = pd.read_pickle(file_path)
            d_list = []
            for d in data_point:
                d = d.drop("rh",axis=1)
                d= d.drop("temp",axis=1)
                if folder == "Multi-Component":
                    if len(d) == 5:
                        d = d[1:5]
                d_list.append(d)
            index = file_name.rfind("\\")
            shorter_file_name = file_name[index + 1:]
            split = shorter_file_name.split("_", 2)
            name = split[0] + "_" + split[1]
            train = d_list
            if data is None:
                data = train
            else:
                data = np.concatenate((data, train))
            if classes is None:
                classes = np.full(len(train), name)
            else:
                classes = np.append(classes, np.full(len(train), name))
    classes = map_classes(classes)
    # Scale data?
    data = data.reshape(data.shape[0], -1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    print(len(data))
    X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=.1)
    return X_train, X_test, y_train, y_test


def results_by_category(test_data, category,model):
    """
    Get classification results by aerosol category

    Args:
        test_data (numpy.ndaray): Data to test model on
        category (str): Name of category to get accuracy for
        model (keras.engine.training.Model): NN to test data with

    Returns:
        None
    """
    # Read in some of the data
    scaler = StandardScaler()
    pred_accuracy = {}
    total_preds = 0
    total_corr = 0
    for key in test_data.keys():
        if category in key:
            split = key.split("_", 2)
            correct_cat = split[0] + "_" + split[1]
            print(test_data[key][0])
            cat_data = np.asarray(test_data[key])
            cat_data = cat_data.reshape(cat_data.shape[0], -1)
            # scale data
            # THIS COULD BE THE PROBLEM
            scaled = scaler.fit_transform(cat_data)
            preds = model.predict(scaled)
            correct_preds = np.count_nonzero(preds == correct_cat)
            total_corr += correct_preds
            total_preds += len(preds)
            print(preds)
            print(correct_cat)
            if correct_preds == 0:
                acc = 0
            else:
                acc = correct_preds/len(preds)
            print(f"PREDICTION ACCURACY for {key} : {acc}")
            if correct_cat not in pred_accuracy.keys():
                pred_accuracy[correct_cat] = 0.
    print(f"OVERALL ACCURACY: {total_corr/total_preds}")
    return

def map_classes(y):
    """
    Map string classes to integers

    Args:
        y (numpy.ndarray): Numpy array of classes as strings

    Returns:
        numpy.ndarray: Numpy array of classes as ints
    """
    dummies = pd.get_dummies(y)
    print(dummies.columns)
    return dummies.values.argmax(1)

def map_test(y,category_name):
    """
    Map Property Derived string classes ints

    y (numpy.ndarray): Numpy array of classes as strings
    category_name (str): Name of categories in y

    Returns:
         numpy.ndarray: Numpy array of classes as ints
    """
    if category_name =="NvHk_NvHk":
        return np.ones(len(y)) * 0
    elif category_name =="NvHk_VHk":
        return np.ones(len(y)) * 1
    elif category_name =="NvHk_VLk":
        return np.ones(len(y)) * 2
    elif category_name =="NvLk_NvHk":
        return np.ones(len(y)) * 3
    elif category_name =="NvLk_NvLk":
        return np.ones(len(y)) * 4
    elif category_name =="NvLk_VHk":
        return np.ones(len(y)) * 5
    elif category_name =="NvLk_VLk":
        return np.ones(len(y)) * 6
    elif category_name =="VHk_VHk":
        return np.ones(len(y)) * 7
    elif category_name =="VLk_VHk":
        return np.ones(len(y)) * 8
    elif category_name =="VLk_VLk":
        return np.ones(len(y)) * 9
    return -99


def class_name_to_int(category_names):
    """
    Map Property Derived string classes ints for classification viz

    Args:
        category_names (List[str]): List of classes as strings

    Returns:
         List[int]: Numpy array of classes as ints
    """
    int_list = np.zeros(len(category_names))
    for i in range(len(category_names)):
        category_name = category_names[i]
        if category_name =="NvHk_NvHk":
            int_list[i] = 0
        elif category_name =="NvHk_VHk":
            int_list[i] = 1
        elif category_name =="NvHk_VLk":
            int_list[i] = 2
        elif category_name =="NvLk_NvHk":
            int_list[i] = 3
        elif category_name =="NvLk_NvLk":
            int_list[i] = 4
        elif category_name =="NvLk_VHk":
            int_list[i] = 5
        elif category_name =="NvLk_VLk":
            int_list[i] = 6
        elif category_name =="VHk_VHk":
            int_list[i] = 7
        elif category_name =="VLk_VHk":
            int_list[i] = 8
        elif category_name =="VLk_VLk":
            int_list[i] = 9
    return int_list

def int_to_class_name(category_names):
    """
    Map Property Derived ints classes strings for classification viz

    Args:
        category_names (List[int]): List of classes as ints

    Returns:
         List[str]: Numpy array of classes as strings
    """
    name_list = []
    for i in range(len(category_names)):
        category_name = category_names[i]
        if category_name == 0:
            name_list.append("NvHk_NvHk")
        elif category_name ==1:
            name_list.append("NvHk_VHk")
        elif category_name ==2:
            name_list.append("NvHk_VLk")
        elif category_name ==3:
            name_list.append("NvLk_NvHk")
        elif category_name ==4:
            name_list.append("NvLk_NvLk")
        elif category_name ==5:
            name_list.append("NvLk_VHk")
        elif category_name ==6:
            name_list.append("NvLk_VLk")
        elif category_name ==7:
            name_list.append("VHk_VHk")
        elif category_name ==8:
            name_list.append("VLk_VHk")
        elif category_name ==9:
            name_list.append("VLk_VLk")
    return np.asarray(name_list)

def int_to_actual(category_names):
    """
    Map Compound Derived ints classes strings for classification viz

    Args:
        category_names (List[int]): List of classes as ints

    Returns:
         List[str]: Numpy array of classes as strings
    """
    name_list = []
    for i in range(len(category_names)):
        category_name = category_names[i]
        if category_name == 0:
            name_list.append("BB_BB")
        elif category_name ==1:
            name_list.append("D_BB")
        elif category_name ==2:
            name_list.append("D_D")
        elif category_name ==3:
            name_list.append("D_S")
        elif category_name ==4:
            name_list.append("D_SS")
        elif category_name ==5:
            name_list.append("SS_BB")
        elif category_name ==6:
            name_list.append("SS_SS")
        elif category_name ==7:
            name_list.append("S_BB")
        elif category_name ==8:
            name_list.append("S_S")
        elif category_name ==9:
            name_list.append("S_SS")
    return np.asarray(name_list)

def property_estimation(folder, target_prop):
    """
    Gets data for property estimation regression

    Args:
        folder (str): Name of folder
        target_prop (int): Index of target property

    Returns:
        X_train (numpy.ndarray): Training X data
        X_test (numpy.ndarray): Test X data
        y_train (numpy.ndarray): Training target data
        y_test (numpy.ndarray): Test target data
    """
    cwd = os.path.dirname(os.path.abspath(os.getcwd()))
    file_extension = "*.pkl"
    directory_path = cwd + "/OpcSimResearch/Aerosol_Category_Outputs/slurm/" + folder
    # Read in data from each json as a dataframe
    data = None
    classes = None
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # if "vol_0.1" in file_name:
            #    continue
            data_point = pd.read_pickle(file_path)
            d_list = []
            y_list = []
            for d in data_point:
                if folder == "Time Averaged":
                    d = d.drop("rh", axis=1)
                    d = d.drop("temp", axis=1)
                    y_list.append(d[target_prop])
                    d = d.drop(target_prop,axis=1)
                if folder == "Multi-Component":
                    if len(d) == 5:
                        d = d[1:5]
                d_list.append(d)
            train = d_list
            if data is None:
                data = train
                classes = y_list
            else:
                data = np.concatenate((data, train))
                classes = np.concatenate(classes,y_list)
    X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=.1)
    return X_train, X_test, y_train, y_test

def read_all_from_db():
    """
    Read all data from pg database and split for downstream ML

    Returns:
        X_train (numpy.ndarray): Training X data
        X_test (numpy.ndarray): Test X data
        y_train (numpy.ndarray): Training target data
        y_test (numpy.ndarray): Test target data
        aero_train (numpy.ndarray): Indices of aerosol referenced in X and y
        aero_val (numpy.ndarray): Indices of aerosol referenced in X and y
    """
    config = configparser.ConfigParser()
    config.read('database.ini')

    host = config.get('postgresql', 'host')
    database = config.get('postgresql', 'database')
    user = config.get('postgresql', 'user')
    password = config.get('postgresql', 'password')
    port = config.get('postgresql', 'port')
    # Read every entry in the experiments column of the opc database
    # and return a list of all the experiments
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password,
        port=port
    )
    cursor = conn.cursor()

    # Get all entries in the experiments table
    cursor.execute("SELECT * FROM experiments")
    experiments = cursor.fetchall()
    # Get the seventh element of each entry in the experiments list
    # (the category name)
    category = []
    data = []
    aerosols = []
    for i in range(len(experiments)):
        cat = experiments[i][6]
        if cat == "NvLk_NvHk":
            cat = "NvHk_NvLk"
        category.append(cat)
        # In the bin_counts table
        # Grab each row with elements of id [0-5]
        # Read it as a dataframe
        # Drop rh and temp columns
        # add it to data list
        df = None
        for j in range(1,6):
            if experiments[i][j] % 1000 == 0:
                print(f"BIN ID: {experiments[i][j]}")
            cursor.execute("SELECT * FROM bin_counts WHERE bin_id=%s", (experiments[i][j],))
            flows = cursor.fetchall()
            if df is None:
                df = pd.DataFrame(flows)
                df = df.drop(0,axis=1)
                df = df.drop(1, axis=1)
                df = df.drop(2, axis=1)
            else:
                df.loc[len(df)] = flows[0][3:]
        df=df.drop(1)
        df=df.drop(3)
        data.append(df)
        # In the aerosol_experiments table
        # Get all aerosols that correspond to the experiment id
        # Make list results a list
        # Append that list to the aerosols list
        cursor.execute("SELECT * FROM aerosol_experiments WHERE experiment_id=%s", (experiments[i][0],))
        results = cursor.fetchall()
        if len(results) != 2:
            print("BROKEN")
            print(f"RESULTS AT: {experiments[i][0]}")
            print(results)
            return
        # For each in results
        res_list = []
        for res in results:
            res_list.append(res[0])
        aerosols.append(res_list)
    conn.close()
    print(len(data))
    if len(data) != len(experiments) or len(aerosols) != len(experiments):
        print("Error: Data and experiments are not the same length")
        return None
    else:
        data = np.array(data)
        data = data.reshape(data.shape[0], -1)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        category = map_classes(np.array(category))
        X_train, X_val, y_train, y_val, aero_train, aero_val = train_test_split(data,category, np.array(aerosols))
        return X_train, X_val, y_train, y_val, aero_train, aero_val

def get_parameter_estimation_classes(aero_train,aero_val, parameter_index):
    """
    Get Aerosols from aerosol table using their PK then grab a specific property by index

    Args:
        aero_train (numpy.ndarray): Array of aerosol_ids
        aero_val (numpy.ndarray): Array of aerosol_ids
        parameter_index (int): Index of parameter of interest

    Returns:
        y_train (numpy.ndarray): Property values of interest
        y_val (numpy.ndarray): Property values of interest
    """
    # ys will be the true kappas
    config = configparser.ConfigParser()
    config.read('database.ini')

    host = config.get('postgresql', 'host')
    database = config.get('postgresql', 'database')
    user = config.get('postgresql', 'user')
    password = config.get('postgresql', 'password')
    port = config.get('postgresql', 'port')
    # Read every entry in the experiments column of the opc database
    # and return a list of all the experiments
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password,
        port=port
    )
    c = conn.cursor()
    y_train = []
    y_val = []
    for aero in aero_train:
        # Get aerosols from aerosol db by id
        # Get aerosol by id
        temp = []
        c.execute("SELECT * FROM aerosol WHERE aerosol_id=%s", (int(aero[0]),))
        temp.append(c.fetchone()[parameter_index])
        c.execute("SELECT * FROM aerosol WHERE aerosol_id=%s", (int(aero[1]),))
        temp.append(c.fetchone()[parameter_index])
        y_train.append(temp)
    for aero in aero_val:
        # Get aerosols from aerosol db by id
        # Get aerosol by id
        temp = []
        c.execute("SELECT * FROM aerosol WHERE aerosol_id=%s", (int(aero[0]),))
        temp.append(c.fetchone()[parameter_index])
        c.execute("SELECT * FROM aerosol WHERE aerosol_id=%s", (int(aero[1]),))
        temp.append(c.fetchone()[parameter_index])
        y_val.append(temp)
    return np.array(y_train), np.array(y_val)

