# pylint: disable=invalid-name, redefined-outer-name
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
from features import Preprocessing


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


def gaussian(x, x0, sigma, a):
    """
    Gaussian function to use in fit.
    """
    return (
        a * (1 / np.sqrt(np.pi * sigma**2)) * np.exp(-((x - x0) ** 2) / (sigma**2))
    )


def sumgaussian(x, x0, x1, sigma0, sigma1, a, b):
    """
    Sum of two gaussian function to possibly use in fit.
    """
    return gaussian(x, x0, sigma0, a) + gaussian(x, x1, sigma1, b)


class Outliers:
    """Class identifying outliers.

    Parameters
    ----------

    """

    def __init__(self, X_train):
        """
        Constructur
        """
        self.X_train = X_train
        self.model = self.make_autoencoder()

    def __call__(self, epochs, nbins, plot=True):
        """Short summary.

        Parameters
        ----------
        epochs : type
            Description of parameter `epochs`.
        nbins : type
            Description of parameter `nbins`.
        plot : type
            Description of parameter `plot`.

        Returns
        -------
        type
            Description of returned object.

        """

        self.fit_autoencoder(epochs)
        indexes = self.outliers(nbins)
        if plot is True:
            self.plot_distribution(self.X_train, indexes, "AGE_AT_SCAN")
        clean = self.clean_dataframe(self.X_train, indexes)
        return clean

    def make_autoencoder(self):
        """Autoencoder trained comparing the output vector with the input features,
        using the Mean Squared Error (MSE) as loss function..

        Parameters
        ----------

        Returns
        -------
        model : type
            The trained model.

        history : type
            summary of how the model trained (training error, validation error).

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

    def fit_autoencoder(self, epochs):
        """Short summary.

        Parameters
        ----------
        epochs : type
            Description of parameter `epochs`.

        Returns
        -------
        type
            Description of returned object.

        """
        history = self.model.fit(
            self.X_train,
            self.X_train,
            validation_split=0.4,
            epochs=epochs,
            batch_size=50,
            verbose=1,
        )
        return history

    def outliers(self, nbins):
        """Identifies ouliers using autoencoder.

        Parameters
        ----------
        nbins : type
            Description of parameter `nbins`.

        Returns
        -------
        type
            Description of returned object.

        """
        x_pred = self.model.predict(self.X_train)

        test_mae_loss = np.mean(
            np.abs(x_pred - np.array(self.X_train)), axis=1
        ).reshape((-1))

        # Plot_test_variables
        d_1 = (test_mae_loss.max() - test_mae_loss.min()) / nbins
        xdiscrete_1 = np.linspace(
            test_mae_loss.min() + d_1 / 2, test_mae_loss.max() - d_1 / 2, nbins
        )
        # Plot_test
        plt.figure(1)
        n_1, _, _ = plt.hist(
            x=test_mae_loss, bins=nbins, color="lightskyblue", label="Sample counts"
        )
        plt.title("Mean Absolute Error Loss", fontsize=24)
        plt.xlabel("Test MAE Loss", fontsize=18)
        plt.ylabel("Number of Samples", fontsize=18)

        # fit as a gaussian
        p0 = [0.4, 0.05, 20]
        # p0 = [8000, 1000, 10]
        fit, fitCov = curve_fit(gaussian, xdiscrete_1, n_1, p0=p0)
        fit_err = np.sqrt(abs(np.diag(fitCov)))
        print(
            f" Test fit parameters: \n x_0 = {fit[0] : .3f} +-{fit_err[0] : .3f}\n sigma_0 = {fit[1] : .3f} +-{fit_err[1] : .3f}\n A = {fit[2] : .3f} +-{fit_err[2] : .3f}\n"
        )
        # Plot fit
        plt.plot(
            np.linspace(test_mae_loss.min(), test_mae_loss.max(), 1000),
            gaussian(np.linspace(test_mae_loss.min(), test_mae_loss.max(), 1000), *fit),
            color="red",
            linewidth=1,
            label="fit",
        )
        # fit a sum of gaussians
        # p1 = [0.4, 24, 0.5, 0.1, 20, 5]
        # fit, fitCov_1 = scipy.optimize.curve_fit(sumgaussian, xdiscrete_1, n_1, p0=p1)
        # fit_err = np.sqrt(abs(np.diag(fitCov_1)))
        # print(
        #    f" First fit parameters: \n x_0 = {fit[0] : .3f} +-{fit_err[0] : .3f}\n x_1 = {fit[1] : .3f} +-{fit_err[1] : .3f}\n sigma_0 = {fit[2] : .3f} +-{fit_err[2] : .3f}\n sigma_1 = {fit[3] : .3f} +-{fit_err[3] : .3f}\n A = {fit[4] : .3f} +-{fit_err[4] : .3f}\n B = {fit[5] : .3f} +-{fit_err[5] : .3f}\n"
        # )
        # Plot_fit_test
        # plt.plot(xcont_1, sumgaussian(xcont_1, *fit), color = 'red', linewidth=1, label='fit')
        plt.show()

        def condition(x, fit):
            if x >= (fit[0] + 3 * fit[1]):
                # return bool(x >= (fit[0] + 3 * fit[1]))
                return 1
            if x <= (fit[0] - 3 * fit[1]):
                # return bool(x <= (fit[0] - 3 * fit[1]))
                return 1
            else:
                return 0
            return False

        final_outliers = [condition(x, fit) for x in test_mae_loss]
        indexes = list(np.flatnonzero(final_outliers))
        print("Number of final outlier samples: ", np.sum(final_outliers))
        print(
            "Percentage of outliers: ",
            np.sum(final_outliers) / len(self.X_train) * 100,
            "%",
        )
        print("Reconstruction MAE error threshold: {} ".format(np.max(test_mae_loss)))
        return indexes

    def plot_distribution(self, dataframe, indexes, feature):
        """Short summary.

        Parameters
        ----------
        dataframe : type
            Description of parameter `dataframe`.
        indexes : type
            Description of parameter `indexes`.
        feature : type
            Description of parameter `feature`.

        Returns
        -------
        type
            Description of returned object.

        """
        y = dataframe.iloc[indexes]
        bins = int(max(y[feature]) - min(y[feature]))
        plt.figure(2)
        n_1, _, _ = plt.hist(
            x=y[feature],
            bins=bins,
            facecolor="lightskyblue",
            edgecolor="blue",
            linewidth=0.5,
            label="Sample counts",
        )
        plt.title("Age Distribution of Outliers", fontsize=24)
        plt.xlabel("Age(years)", fontsize=18)
        plt.ylabel("Number of Samples", fontsize=18)
        return plt.show()

    def clean_dataframe(self, dataframe, indexes):
        """Short summary.

        Parameters
        ----------
        dataframe : type
            Description of parameter `dataframe`.
        indexes : type
            Description of parameter `indexes`.

        Returns
        -------
        type
            Description of returned object.

        """
        y = dataframe.iloc[indexes]
        clean = dataframe.drop(index=y.index)
        return clean


if __name__ == "__main__":
    prep = Preprocessing()
    df = prep.read_file("data/FS_features_ABIDE_males.csv")
    df = prep(df, "raw", False)
    df_AS, df_TD = prep.split_file(df)
    out = Outliers(df_TD)
    clean_dataframe = out(epochs=100, nbins=500, plot=True)
