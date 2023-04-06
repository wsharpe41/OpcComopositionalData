import math

from Reverse_Simulator.aerosol_category import AerosolCategory
from Reverse_Simulator import classify_aerosol, property_nn
import numpy as np
import pandas as pd
import os
from tensorflow.keras import layers, models
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_dataset():
    """
    Generate testing dataset for parameter estimation task

    Returns:
         None
    """
    # Need to generate example aerosol response keeping size in a consistent range
    # I think I need particle size in input so 13, instead of 12 parameters for median size
    # Vary HVap from 80 to 300
    # Vary SVp @ 450 K from 0 to 5e-10
    # Vary Molar Mass from 80 to 200
    # Vary Density from 0.8 to 2.2
    # Vary Kappa from 0 to 1.1
    svp_range = [0]
    for i in range(8,31):
        svp_range.append(10**-i)
    # Made it so everything needs same length
    aerosol = AerosolCategory(diameter_range=np.arange(.1, 1, .01),
                              geometric_standard_deviation_range=np.arange(1.2, 1.6, .01),
                              temp_range=[450, 298, 450], rh_range=[20, 90, 90],
                              kappa_range=[0.,0.,0.],
                              svp=[0,10**-13,10**-15], svp_ref=298, mm_range=[120.,120.,120.], hvap_range=[200.,200.,200.],
                              density_range=[1.1,1.1,1.1])
    aerosol.produce_category_outputs(120,"Svp_Parameter_Estimation",experimental_length=600,mode="Parameter Estimation")
    # Need to link each of these with their respective parameters
    return

def read_parameter_dataset():
    """
    Read test parameter dataset

    Returns:
        opc_readings (numpy.ndarray): OPC data
        param_list (numpy.ndarray): List of parameters for parameter estimation
    """
    cwd = os.path.dirname(os.path.abspath(os.getcwd()))
    data = pd.read_pickle(cwd + "/OpcSimResearch/Aerosol_Category_Outputs/Parameter_Estimation/Svp_Parameter_Estimation/Svp_Parameter_Estimation_category_examples.pkl")
    opc_readings = np.asarray(data[0])
    param_list = []
    parameters = data[1]
    for parameter in parameters:
        param_list.append(list(parameter.values()))
    return opc_readings.reshape(opc_readings.shape[0], -1), np.asarray(param_list)

def normalize_svp(params):
    """

    :param params:
    :return:
    """
    # Zero should be the lowest not the highest
    df = pd.DataFrame(params)
    df[4] = df[4].apply(lambda x: math.log10(x) if x > 0 else -31.)
    return np.asarray(df)


def parameter_estimation_from_db(parameter_index):
    """

    :param parameter_index:
    :return:
    """
    # Read in all data regardless and estimate kappa
    X_train, X_val, y_train, y_val, aero_train, aero_val = classify_aerosol.read_all_from_db()
    y_train, y_val = classify_aerosol.get_parameter_estimation_classes(aero_train,aero_val,parameter_index)

    #classify_aerosol.classify_with_automl(X_train,y_train)
    input_dim = X_train[0].shape
    model = parameter_nn(input_dim)
    esc = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                  loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])
    history = model.fit(x=X_train, y=y_train, epochs=500, batch_size=8, validation_data=(X_val, y_val), callbacks=[esc])
    print(history.history.keys())
    plt.plot(history.history['val_loss'])
    plt.show()
    predictions = (model.predict(X_val))
    print(predictions[0:10])
    print(y_val[0:10])
    rmse = np.sqrt(np.mean((predictions - y_val) ** 2))
    print('RMSE:', rmse)
    return model

def parameter_nn(input_dims):
    """
    Parameter Estimation Neural Net

    Args:
        input_dims: Shape of input

    Returns:
         keras.engine.training.Model: Parameter estimation model
    """
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_dims))
    model.add(layers.Flatten())
    model.add(layers.Dense(96, name="Dense1"))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, name="Dense2"))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64, name="Dense3"))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(32, name="Dense4"))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(2))
    #model.add(layers.Linear())
    model.summary()
    return model