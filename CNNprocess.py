"""CSC110 Final Project 2020

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
from util import *
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten
import sklearn.metrics as sm
from scipy.stats.stats import pearsonr, spearmanr
from typing import Tuple
from visualization import *


class CNNProcess:
    """ A class representing the entire process of CNN

    Instance Attribute:
        - window_size: prediction period
        - n_step: given a time series sequence, previous n_step's
                  are grouped to a vector as input
    """
    window_size: int
    n_step: int

    def __init__(self, window_size: int, n_step: int) -> None:
        """ Initialize a CNN process object
        """
        self.window_size = window_size
        self.n_step = n_step

    def load_data(self, data: pd.DataFrame) -> None:
        self.data = data

        self.X_train, self.X_test, self.y_train, self.y_test = preprocess(self.data['GMSL'].values,
                                                                          timestep=self.n_step,
                                                                          scalar='norm', randState=20)
        # self.X_train, self.X_test, self.y_train, self.y_test = preprocess(self.data['GMSL'].values[:len(self.data)
        # - self.n_step - self.window_size], timestep=self.n_step, scalar='norm', randState=20)

    def train_model(self) -> None:
        """ Initialize the CNN model from Sequential and train the model
        """
        self.n_feature = 1
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], self.n_feature)

        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(self.n_step, self.n_feature)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(20, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

        self.model.fit(self.X_train, self.y_train, epochs=100, verbose=False)

    def test_data(self) -> None:
        """ Validate the test data
        """
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], self.n_feature)

        y_test_pred = self.model.predict(self.X_test)

        # result stats
        statistics(self.y_test, y_test_pred)

    def predict_result(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Predict the GSML (window size is specified as instance attribute)
            Prediction Method 1 : test prediction
        """
        x = self.data['GMSL'].values
        min_GMSL, max_GMSL = np.min(x), np.max(x)
        x_norm = (self.data['GMSL'].values[-(self.window_size + self.n_step):] - min_GMSL) / (max_GMSL - min_GMSL)
        x_act, y_act = input_generator_2D(x_norm, self.n_step)

        x_act = x_act.reshape(x_act.shape[0], x_act.shape[1], self.n_feature)
        y_pred = self.model.predict(x_act)
        y_test, y_test_pred = y_act, y_pred

        y_test_pred = y_test_pred.reshape(1, self.window_size).ravel()

        y_test = y_test * (max_GMSL - min_GMSL) + min_GMSL
        y_test_pred = y_test_pred * (max_GMSL - min_GMSL) + min_GMSL

        print("Pearson correlation =", round(pearsonr(y_test, y_test_pred)[0], 4))
        print("Spearsman correlation =", round(spearmanr(y_test, y_test_pred)[0], 4))
        print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 4))
        print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 4))
        print("Root Mean squared error =", round(np.sqrt(sm.mean_squared_error(y_test, y_test_pred)), 4))
        print("MAPE =", round(np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100))

        return y_test, y_test_pred

    def visualization(self, actual, pred) -> None:
        """Visualize the test prediction using comparison graph"""
        plt = plot_comparison_graph(pred, actual, 'Number of Month in advance of Dec 2013',
                                    'Sea Level Variation (mm)', 'Convolutional Neural Network ')
        plt.show()

    def pred_validation(self, pred_period: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns a tuple of numpy arrays representing predicted and actual data
            Prediction Method 2: predict future data
        """
        pred_list = []
        min_GMSL, max_GMSL = np.min(self.data['GMSL']), np.max(self.data['GMSL'])
        data = normalize(self.data['GMSL'].values)[:]
        # data = data[:-pred_period]

        for _ in range(pred_period):
            self.x, self.y = input_generator_2D(data, self.n_step)

            n_feature = 1
            self.x = self.x.reshape(self.x.shape[0], self.x.shape[1], n_feature)

            model = Sequential()
            model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(self.n_step, n_feature)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(25, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')

            model.fit(self.x, self.y, epochs=70, verbose=False)

            self.x_test = data[-self.n_step:]
            self.x_test = self.x_test.reshape(1, len(self.x_test), n_feature)
            y_pred = model.predict(self.x_test)
            y_prediction = y_pred[0, 0] * (max_GMSL - min_GMSL) + min_GMSL
            pred_list.append(y_prediction)

            data = np.append(data, y_pred[0, 0])

        return np.array(pred_list), self.data['GMSL'].values[:]
