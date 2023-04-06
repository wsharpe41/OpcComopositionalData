import configparser
import psycopg2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Reverse_Simulator import classify_aerosol

def read_experiments_from_db(start_index,stop_index):
    """
    Reads experiments from experiment table between two indices and grabs the corresponding bin_counts and aerosols

    Args:
        start_index (int): First index in experiments table
        stop_index (int): Second index in experiments table

    Returns:
        X_train (numpy.ndarray) - Training bin_counts values
        X_test (numpy.ndarray) - Testing bin_counts values
        y_train (numpy.ndarray) - Training aerosol categories
        y_test (numpy.ndarray) - Testing aerosol categories
        aero_val (numpy.ndarray) - Training aerosols
        aero_train (numpy.ndarray) - Testing aerosols
        rh_val (numpy.ndarray) - Training rh values from bin_counts
        rh_train (numpy.ndarray) - Testing rh values from bin_counts
        temp_val (numpy.ndarray) - Training temp values from bin_counts
        temp_train (numpy.ndarray) - Testing temp values from bin_counts
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
    cursor.execute(f"SELECT * FROM experiments WHERE experiment_id BETWEEN {start_index} AND {stop_index}")
    experiments = cursor.fetchall()
    # Get the seventh element of each entry in the experiments list
    # (the category name)
    category = []
    data = []
    aerosols = []
    rhs = []
    temps = []
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
        flow_rhs = []
        flow_temps = []
        for j in range(1,6):
            if experiments[i][j] % 1000 == 0:
                print(f"BIN ID: {experiments[i][j]}")
            cursor.execute("SELECT * FROM bin_counts WHERE bin_id=%s", (experiments[i][j],))
            flows = cursor.fetchall()
            flow_rhs.append(flows[0][1])
            flow_temps.append(flows[0][2])
            if df is None:
                df = pd.DataFrame(flows)
                df = df.drop(0,axis=1)
                df = df.drop(1, axis=1)
                df = df.drop(2, axis=1)
            else:
                df.loc[len(df)] = flows[0][3:]
        data.append(df)
        rhs.append(flow_rhs)
        temps.append(flow_temps)
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
        category = classify_aerosol.map_classes(np.array(category))
        X_train, X_val, y_train, y_val, aero_train, aero_val,rh_train,rh_val,temp_train,temp_val = train_test_split(data,category, np.array(aerosols),rhs,temps)
        return X_train, X_val, y_train, y_val, aero_train, aero_val,rh_train,rh_val,temp_train,temp_val