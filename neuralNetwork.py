"""CSC110 Final Project 2020: preprocess.py

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
from util import *
from typing import Tuple


class NeuralNetwork:
    """ A class representing our neural network model from scratch
    Instance Attributes:
        - ...
    """

    def __init__(self) -> None:
        """ Initialize a Neural Network object
        """
        self.inputSize = 3
        self.hiddenSize = 2
        self.outputSize = 1

        # initialize weights and bias to be small random numbers
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
        self.b1 = np.random.randn(self.hiddenSize)
        self.b2 = np.random.randn(self.outputSize)
        self.learning_rate = 0.002

    def sigmoid(self, a: np.ndarray) -> np.ndarray:
        """ Returns a number between 0 and 1 by applying sigmoid activation function on the input <a>
        """
        print(type(a), type(1 / (1 + np.exp(-a))))
        return 1 / (1 + np.exp(-a))

    def forward_propagation(self, x) -> Tuple[np.ndarray, np.ndarray]:
        """ Propagate inputs <x> through network
            Returns a tuple where the first element representing hidden weighted input
            and the second element being the final output.
            
            This is done by the following procedure
            Z = x dot product W1 + bias term (b1)
            Z = Sigmoid Z
            Y = Z dot product W2 + bias term (b2)
        """
        Z = self.sigmoid(dot_product(x, self.W1) + self.b1)
        y_hat = dot_product(Z, self.W2) + self.b2
        y_hat = np.array([y for l in y_hat for y in l])

        return Z, y_hat

    def cost(self, y_true: np.ndarray, y_est: np.ndarray) -> float:
        """ Returns the cost function of the actual value <y_true>
            and the predicted value <y_est>
        """
        return (0.5 * (y_true - y_est) ** 2).sum()

    def delta_W1(self, X: np.ndarray, Z: np.ndarray, T: np.ndarray, Y: np.ndarray, W2: np.ndarray) -> np.ndarray:
        """ Return the change of hidden weights that will be applied from input to hidden neuron
        """
        # iterative method
        #     w = np.zeros((D, M))
        #     for t in range(n):
        #         for h in range(M):
        #             for j in range(D):
        #                 w[j, h] += (np.array(T)[t] - Y[t])*W2[h,0]*Z[t,h]*(1-Z[t,h])*X[t,j]

        w = -X.T.dot((T - Y).reshape(-1, 1).dot(W2.T) * Z * (1 - Z))

        return w

    def delta_W2(self, Z: np.ndarray, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """ Return the change of output weight that will be applied from hidden neuron to output neuron
        """
        v = -Z.T.dot(T - Y).reshape(-1, 1)

        # iterative method
        #     n = T.shape[0]
        #     v = np.zeros((M,K))
        #     for t in range(n):
        #         for h in range(M):
        #             v[h, 0] += (np.array(T)[t] - Y[t])*Z[t,h]

        return v

    def delta_b1(self, Z: np.ndarray, T: np.ndarray, Y: np.ndarray, W2: np.ndarray) -> np.ndarray:
        """ Return the change of hidden bias that will be applyed from input to hidden neuron
        """
        return -((T - Y).reshape(-1, 1).dot(W2.T) * Z * (1 - Z)).sum(axis=0)

    def delta_b2(self, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """ Returns the change of output bias that will be applied from hidden neuron to output neuron
        """
        return -(T - Y).sum()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Given <X>, return model's prediction
        """
        return self.forward_propagation(X)[1]
