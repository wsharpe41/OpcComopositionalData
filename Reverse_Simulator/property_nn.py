from abc import ABC

from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import optuna
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Reverse_Simulator import classify_aerosol


def keras_cnn_model(input_dims,category_dims=10):
    """
    Create simple model using keras

    Args:
        input_dims: Dimension of input
        category_dims (int): Number of output categories

    Returns:
        keras.engine.training.Model: Simple NN
    """
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_dims))
    model.add(layers.Flatten())
    model.add(layers.Dense(96, name="Dense1"))
    model.add(layers.ReLU())

    model.add(layers.Dense(64, name="Dense2"))
    model.add(layers.ReLU())

    model.add(layers.Dense(category_dims))
    model.add(layers.Softmax())
    model.summary()
    return model

def train_model(X_train,y_train,X_val,y_val):
    """
    Train NN

    Args:
        X_train (numpy.ndarray): Training X data
        X_val (numpy.ndarray): Val X data
        y_train (numpy.ndarray): Training target data
        y_val (numpy.ndarray): Val target data

    Returns:
        keras.engine.training.Model: Simple NN
    """
    model = keras_cnn_model(np.shape(X_train[0]))
    print(np.shape(X_train[0]))
    esc = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    history = model.fit(x=X_train,y=y_train, epochs=500,batch_size=8, validation_data=(X_val,y_val),callbacks=[esc])
    print(history.history.keys())
    return model

def test_model(model, X_test, y_test):
    """
    Test NN on aerosol outputs

    Args:
        model (keras.engine.training.Model): Simple NN
        X_test (numpy.ndarray): Testing data
        y_test (numpy.ndarray): Classes for data

    Returns:
        None
    """
    predictions = (model.predict(X_test))
    most_likely = np.argmax(predictions,axis=1)
    test_acc = tf.keras.metrics.Accuracy()
    test_acc.update_state(y_test,most_likely)
    y_test = classify_aerosol.int_to_class_name(y_test)
    most_likely = classify_aerosol.int_to_class_name(most_likely)
    #y_test = classify_aerosol.int_to_actual(y_test)
    #most_likely = classify_aerosol.int_to_actual(most_likely)

    class_labels = ["NvHk_NvHk","NvHk_VHk","NvHk_VLk","NvLk_NvHk","NvLk_NvLk","NvLk_VHk",
                                         "NvLk_VLk","VHk_VHk","VLk_VHk","VLk_VLk"]
    #class_labels = ["BB_BB","D_BB","D_D","D_S","D_SS","SS_BB","SS_SS","S_BB","S_S","S_SS"]
    confusion = confusion_matrix(y_test,most_likely,normalize='true',
                                 labels=class_labels
                                 )
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion,
                                  display_labels=class_labels
                                  )
    disp.plot()
    plt.title("Property Derived Confusion")
    plt.xticks(rotation=30)
    plt.show()
    print(f" TEST ACCURACY: {test_acc.result()}")


# Write NN to be an optuna problem
def create_model(trial):
    """
    Create simple NN

    Args:
        trial: Optuna trial

    Returns:
        keras.engine.training.Model: NN Model
    """
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(5,24)))
    model.add(layers.Flatten())
    n_dense = trial.Int('n_dense',1,5,step=1)
    for i in range(n_dense):
        model.add(layers.Dense(units=trial.Int('dense_'+str(i),16,128,step=16), name="Dense"+str(i)))
        model.add(layers.ReLU())
    model.add(layers.Dense(10))
    model.add(layers.Softmax())
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(trial.Choice('learning_rate',values=[1e-2,1e-3,1e-4,1e-5])),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    return model

def optimize_hyperparameters(X_train,y_train,X_val,y_val):
    """
    Optimize NN hyperparameters

    Args:
        X_train (numpy.ndarray): Training X data
        X_val (numpy.ndarray): Val X data
        y_train (numpy.ndarray): Training target data
        y_val (numpy.ndarray): Val target data

    Returns:
        None
    """
    esc = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    tuner = RandomSearch(create_model,objective='val_accuracy',max_trials=50,executions_per_trial=5,directory="optuna_model_50",project_name="opc_nn")
    tuner.search_space_summary()
    tuner.search(
        X_train,y_train,
        epochs=500,
        validation_data=(X_val,y_val),
        callbacks=[esc],
        batch_size=8
    )
    print(tuner.results_summary())

def test_varied_conditions_model(model, X_test, y_test,rh_test,temp_test):
    """
    Split data points by TD temperatures

    Args:
        temp_test (numpy.ndarray): Numpy Array of aerosol output temperatures
        y_test (numpy.ndarray): Numpy Array of aerosol output targets
        X_test (numpy.ndarray): Numpy Array of aerosol outputs
        rh_test (List[float]): List of rh values
        model (keras.engine.training.Model): NN to test data points with

    Returns:
        None
    """
    test_acc = tf.keras.metrics.Accuracy()
    rhs = split_rh(rh_test,y_test,X_test,temp_test)
    temps = split_temp(temp_test,y_test,X_test,rh_test)
    for rh in rhs.keys():
        value = rhs[rh]
        X = np.array([values[0] for values in value])
        y = np.array([values[1] for values in value])
        predictions = (model.predict(X))
        most_likely = np.argmax(predictions,axis=1)
        test_acc.update_state(y,most_likely)
        print(f" TEST ACCURACY for {rh}: {test_acc.result()}")
    for temp in temps.keys():
        value = temps[temp]
        X = np.array([values[0] for values in value])
        y = np.array([values[1] for values in value])
        predictions = (model.predict(X))
        most_likely = np.argmax(predictions,axis=1)
        test_acc.update_state(y,most_likely)
        print(f" TEST ACCURACY for {temp}: {test_acc.result()}")


def split_rh(rh_test,y_test,X_test,temp_test):
    """
    Split data points by flow RH

    Args:
        temp_test (numpy.ndarray): Numpy Array of aerosol output temperatures
        y_test (numpy.ndarray): Numpy Array of aerosol output targets
        X_test (numpy.ndarray): Numpy Array of aerosol outputs
        rh_test (List[float]): List of rh values

    Returns:
        Dict[str,List[float]]: Split points by rh
    """

    split_rh = {'10_60':[],'10_70':[],'10_80':[],'20_60':[],'20_70':[],'20_80':[],'30_60':[],'30_70':[],'30_80':[]}
    for i in range(len(rh_test)):
        if max(temp_test[i]) < 395.:
            continue
        elif(max(rh_test[i])) > 80.0:
            if min(rh_test[i]) <20:
                split_rh['10_80'].append([X_test[i],y_test[i]])
            elif min(rh_test[i]) > 30:
                split_rh['30_80'].append([X_test[i], y_test[i]])
            else:
                split_rh['20_80'].append([X_test[i], y_test[i]])
        elif(max(rh_test[i])) > 70.0:
            if min(rh_test[i]) <20:
                split_rh['10_70'].append([X_test[i],y_test[i]])
            elif min(rh_test[i]) > 30:
                split_rh['30_70'].append([X_test[i], y_test[i]])
            else:
                split_rh['20_70'].append([X_test[i], y_test[i]])
        elif(max(rh_test[i])) < 70.0:
            if min(rh_test[i]) <20:
                split_rh['10_60'].append([X_test[i],y_test[i]])
            elif min(rh_test[i]) > 30:
                split_rh['30_60'].append([X_test[i], y_test[i]])
            else:
                split_rh['20_60'].append([X_test[i], y_test[i]])
    return split_rh

def split_temp(temp_test,y_test,X_test,rh_test):
    """
    Split data points by TD temperatures

    Args:
        temp_test (numpy.ndarray): Numpy Array of aerosol output temperatures
        y_test (numpy.ndarray): Numpy Array of aerosol output targets
        X_test (numpy.ndarray): Numpy Array of aerosol outputs
        rh_test (List[float]): List of rh values

    Returns:
        Dict[str,List[float]]: Split points by temperature
    """

    split_temp = {'270_355':[],'270_370':[],'270_385':[],'280_355':[],'280_370':[],'280_385':[],'290_355':[],'290_370':[],'290_385':[]}
    for i in range(len(temp_test)):
        if max(rh_test[i]) < 85.:
            continue
        if(max(temp_test[i])) > 385.0:
            if min(temp_test[i]) <280:
                split_temp['270_385'].append([X_test[i],y_test[i]])
            elif min(temp_test[i]) > 290:
                split_temp['290_385'].append([X_test[i], y_test[i]])
            else:
                split_temp['280_385'].append([X_test[i], y_test[i]])
        elif(max(temp_test[i])) > 370.0:
            if min(temp_test[i]) <280:
                split_temp['270_370'].append([X_test[i],y_test[i]])
            elif min(temp_test[i]) > 290:
                split_temp['290_370'].append([X_test[i], y_test[i]])
            else:
                split_temp['280_370'].append([X_test[i], y_test[i]])
        elif(max(temp_test[i])) < 370.0:
            if min(temp_test[i]) <280:
                split_temp['270_355'].append([X_test[i],y_test[i]])
            elif min(temp_test[i]) > 290:
                split_temp['290_355'].append([X_test[i], y_test[i]])
            else:
                split_temp['280_355'].append([X_test[i], y_test[i]])
    return split_temp