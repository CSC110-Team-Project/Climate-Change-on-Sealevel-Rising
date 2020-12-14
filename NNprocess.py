"""CSC110 Final Project 2020: NNprocess.py

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
from preprocess import *
from neuralNetwork import *
import matplotlib.pyplot as plt
from typing import List, Tuple
from copy import deepcopy
from visualization import *


class NNProcess:
    """ A class representing the entire NN process

    Instance Attributes:
        - X_train: independent variable for training
        - y_train: dependent variable to be predicted by model
        - X_test: independent variable for validation
        - y_test: dependent variable to test the validation set
        - X_Predict: independent variable to test accuracy
        - Y_Predict: dependent variable to compare itself to actual data
        - NN: neural network instance
        - period: prediction period
    """

    def __init__(self, period: int) -> None:
        """ Initialize a neural network object
        """
        self.period = period

    def network_training(self, NN, err_threshold=2) -> List[float]:
        """ A helper function that trains <NN> and specifies error threshold
            and returns the cost J at each epoch
        """
        costs = []
        x_train, y_train = self.x_train, self.y_train
        # training for 500 epochs
        for epoch in range(500):
            hidden, output = NN.forward_propagation(x_train)
            loss = NN.cost(y_train, output)
            costs.append(loss)

            if loss < err_threshold:
                print('Trained variables:')
                print('W1: \n', NN.W1)
                print('b1: \n', NN.b1)
                print('W2: \n', NN.W2)
                print('b2: \n', NN.b2)
                break

            NN.W2 -= NN.learning_rate * NN.delta_W2(hidden, y_train, output)
            NN.b2 -= NN.learning_rate * NN.delta_b2(y_train, output)
            NN.W1 -= NN.learning_rate * NN.delta_W1(x_train, hidden, y_train, output, NN.W2)
            NN.b1 -= NN.learning_rate * NN.delta_b1(hidden, y_train, output, NN.W2)
        self.NN = NN
        self.x_train, self.y_train = x_train, y_train
        return costs

    def load_data(self, data) -> None:
        """ Pre-process the input and split data for training and validating
        """
        self.data = data

        self.x_train, self.x_test, self.y_train, self.y_test = preprocess_for_NN(deepcopy(self.data[:-self.period]),
                                                                                 0.2)

    def train_model(self) -> None:
        """ Initialize an instance of NeuralNetwork and trains it
        """
        self.NN = NeuralNetwork()
        costs = self.network_training(self.NN)
        plt.plot(costs)
        plt.show()

    def test_data(self) -> None:
        """ Use the trained model to validate accuracy
        """
        x_test, y_test = self.x_test, self.y_test
        # NN testing
        y_test_pred = self.NN.predict(x_test)

        # result stats
        statistics(y_test, y_test_pred)

        self.x_test, self.y_test = x_test, y_test

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Predict the GSML (period is specified as instance attribute)
            Return a tuple representing actual and predicted data
        """
        inputdata = deepcopy(self.data[-self.period:])
        inputdata['GMSL'], inputdata['Extent'], \
        inputdata['LandAverageTemperature'] = normalize(inputdata['GMSL']), \
                                              normalize(inputdata['Extent']), \
                                              normalize(inputdata['LandAverageTemperature'])

        self.X_act = inputdata[['GMSL', 'Extent', 'LandAverageTemperature']].iloc[:-1, :].values
        self.Y_act = deepcopy(self.data['GMSL'][-self.period + 1:].values)
        delta = self.period * 2
        min_gmsl, max_gmsl = np.min(self.data['GMSL']), np.max(self.data['GMSL'])

        y_test_pred = self.NN.forward_propagation(self.X_act)[1] * ((max_gmsl - min_gmsl) + min_gmsl) + delta

        statistics(self.Y_act, y_test_pred)

        return self.Y_act, y_test_pred

    def visualization(self, test, pred) -> None:
        """ Visualize our prediction
        """
        plt = plot_comparison_graph(pred, test, 'Number of Month in advance of Dec 2013',
                                    'Sea Level Variation (mm)', 'Neural Network')
        plt.show()
