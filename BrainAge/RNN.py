# pylint: disable=invalid-name, redefined-outer-name, import-error
"""
Module implements RNN which tries to replicate given data
"""
import pickle

from scipy.optimize import curve_fit
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Activation
from keras.models import Model
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

import matplotlib.pyplot as plt
import numpy as np

# Custom activation function
from preprocessing import Preprocessing


def step_wise(x, N=4, a=100):
    """Custom step-wise function to use as activation for RNN.

    Parameters
    ----------
    x : array-like
        Description of parameter `x`.
    N : integer
        Number of steps. Default is  4.
    a : integer
        Tuning parameter. Default is 100.
    Returns
    -------
    y : array-like
        Return function.

    """
    y = 1 / 2
    for j in range(1, N):
        y += (1 / (2 * (N - 1))) * (K.tanh(a * (x - (j / N))))
    return y



class RNN:
    """Class implementing peplicator neural network.

    Parameters
    ----------

    """

    def __init__(self, X_train):
        """Constructur.

        Parameters
        ----------
        X_train : datframe-liket
            Dataframe given to train the RNN.

        """


        self.X_train = X_train


    def __call__(self, epochs):
        """Short summary.

        Parameters
        ----------
        epochs : integer
            Epochs needed to train the RNN .

        """
        model=self.make_autoencoder()
        self.fit_autoencoder(model,epochs)

    def make_autoencoder(self):
        """Autoencoder trained comparing the output vector with the input features,
        using the Mean Squared Error (MSE) as loss function..

        Parameters
        ----------

        Returns
        -------
        model : type
            The trained model.
        """
        get_custom_objects().update({"step_wise": Activation(step_wise)})

        inputs = Input(shape=self.X_train.shape[1])
        hidden = Dense(30, activation="tanh")(inputs)
        hidden = Dense(2, activation="step_wise")(hidden)
        hidden = Dense(30, activation="tanh")(hidden)
        outputs = Dense(self.X_train.shape[1], activation="linear")(hidden)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["MAE"])
        model.summary()
        return model

    def fit_autoencoder(self,model, epochs):
        """Short summary.

        Parameters
        ----------
        epochs : type
            Description of parameter `epochs`.

        Returns
        -------
        history : type
            summary of how the model trained (training error, validation error).

        """
        history = model.fit(
            self.X_train,
            self.X_train,
            validation_split=0.4,
            epochs=epochs,
            batch_size=50,
            verbose=1,
        )
        with open(
            "models/autoencoder_pkl", "wb"
        ) as files:
            pickle.dump(model, files)
        return history



if __name__ == "__main__":
    prep = Preprocessing()
    df = prep.read_file("data/FS_features_ABIDE_males.csv")
    df = prep(df, "normalized", False)
    df = prep.remove_strings(df)
    df_AS, df_TD = prep.split_file(df)
    out = RNN(df_TD)
    df_TD = clean_dataframe = out(epochs=200)
