"""CSC110 Final Project 2020: util.py

Copyright and Usage Information
===============================
This file is part of the CSC110 final project: Data Analysis on Rising Sea Level,
developed by Charlie Guo, Owen Zhang, Terry Tu, Vim Du.
This file is provided solely for the course evaluation purposes of CSC110 at University of Toronto St. George campus.
All forms of distribution of this code, whether as given or with any changes, are strictly prohibited.
The code may have referred to sources beyond the course materials, which are all cited properly in project report.
For more information on copyright for this project, please contact any of the group members.

This file is Copyright (c) 2020 Charlie Guo, Owen Zhang, Terry Tu and Vim Du.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
from typing import Tuple


def standardize(arr: np.ndarray) -> np.ndarray:
    """ Returns the standardized data.
        To do this, the original <arr> is being transformed to have a mean value 
        of zero, and std of 1 using the formula 
        new arr = (arr - mean(arr))/ std(arr)
    """
    if arr.ndim == 1:
        arr = (arr - np.mean(arr)) / np.std(arr)
    else:
        for i in range(arr.shape[-1]):
            arr[:, i] = (arr[:, i] - np.mean(arr[:, i])) / np.std(arr[:, i])

    return arr


def normalize(arr: np.ndarray) -> np.ndarray:
    """ Returns the normalized data. 
        To do this, the original <arr> is being rescaled to have values between 0 and 1 using
        the formula new arr = (arr - min(arr)) / (max(arr) - min(arr))
    """
    if arr.ndim == 1:
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    else:
        for i in range(arr.shape[-1]):
            arr[:, i] = (arr[:, i] - np.min(arr[:, i])) / (np.max(arr[:, i]) - np.min(arr[:, i]))

    return arr


# Single Input Series
def input_generator_2D(seq: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    input_x, input_y = [], []

    for i in range(len(seq) - n):
        input_x.append(seq[i:i + n])
        input_y.append(seq[i + n])

    return np.array(input_x), np.array(input_y)


# Multiple Input Series
def input_generator_3D(seq: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    input_x, input_y = [], []

    for i in range(len(seq) - n):
        input_x.append(seq[i:i + n, :])
        input_y.append(seq[i + n, -1])

    return np.array(input_x), np.array(input_y)


def preprocess(inputData: np.ndarray, timestep: int = 3, scalar: str = None, randState: int = 10) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Return the separated training and testing data set
    """
    ndim = inputData.ndim

    if scalar == 'std':
        inputData = standardize(inputData)
    elif scalar == 'norm':
        inputData = normalize(inputData)

    if ndim > 1:
        x, y = input_generator_3D(inputData, timestep)
    else:
        x, y = input_generator_2D(inputData, timestep)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=randState)

    return x_train, x_test, y_train, y_test


def preprocess_for_NN(inputData: np.ndarray, t: int, randState: int = 10) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inputData['GMSL'] = normalize(inputData['GMSL'])
    inputData['Extent'] = normalize(inputData['Extent'])
    inputData['LandAverageTemperature'] = normalize(inputData['LandAverageTemperature'])
    x = inputData[['GMSL', 'Extent', 'LandAverageTemperature']].iloc[:-1, :].values
    y = inputData['GMSL'][1:].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t, random_state=randState)

    return x_train, x_test, y_train, y_test


def helper(v1: np.ndarray, v2: np.ndarray) -> int:
    """ Returns dot product of two vectors <v1> and <v2>
    """
    res = 0
    for i in range(len(v1)):
        res += v1[i] * v2[i]
    return res


def dot_product(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """ Returns a matrix that is computed from m1 dot product m2
    """
    res = []
    for i in range(len(m1)):
        col = []
        for j in range(len(m2[0])):
            col.append(helper(m1[i], np.array([r[j] for r in m2])))
        res.append(col)
    return np.array(res)


def statistics(act: np.ndarray, pred: np.ndarray) -> None:
    """ Print the stats that describe relationship between
        actual and predicted result
    """
    print("Mean absolute error =", round(sm.mean_absolute_error(act, pred), 4))
    print("Root Mean squared error =", round(np.sqrt(sm.mean_squared_error(act, pred)), 4))
    print("Median absolute error =", round(sm.median_absolute_error(act, pred), 4))
    print("Explain variance score =", round(sm.explained_variance_score(act, pred), 4))
    print("R2 score =", round(sm.r2_score(act, pred), 4))
