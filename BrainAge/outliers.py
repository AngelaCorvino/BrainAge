"""
Module  selects samples with reconstruction error larger than 3 sigmas and removes outlier samples from final dataframe.
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

    def __init__(self,dataframe):
        """
        Constructur
        """
        self.dataframe=dataframe
        self.model=self.model_upload()

    def __call__(self, nbins, plot=True):
        """Short summary.

        Parameters
        ----------

        nbins : type
            Description of parameter `nbins`.
        plot : type
            Description of parameter `plot`.

        Returns
        -------
        type
            Description of returned object.

        """


        indexes = self.outliers(nbins)
        if plot is True:
            self.plot_distribution(indexes, "AGE_AT_SCAN")
        clean = self.clean_dataframe(indexes)
        return clean


    def model_upload(self):
        with open(
            "models/autoencoder_pkl" , "rb"
        ) as f:
            model = pickle.load(f)
        return model


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
        x_pred = self.model.predict(self.dataframe)

        test_mae_loss = np.mean(
            np.abs(x_pred - np.array(self.dataframe)), axis=1
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
            np.sum(final_outliers) / len(self.dataframe) * 100,
            "%",
        )
        print("Reconstruction MAE error threshold: {} ".format(np.max(test_mae_loss)))
        return indexes

    def plot_distribution(self, indexes, feature):
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
        y = self.dataframe.iloc[indexes]
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

    def clean_dataframe(self, indexes):
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
        y = self.dataframe.iloc[indexes]
        clean = self.dataframe.drop(index=y.index)
        return clean


if __name__ == "__main__":
    prep = Preprocessing()
    df = prep.read_file("data/FS_features_ABIDE_males.csv")
    df = prep(df, "normalized", False)
    df = prep.remove_strings(df)
    df_AS, df_TD = prep.split_file(df)
    out = Outliers(df_TD)
    df_TD = clean_dataframe = out(nbins=500, plot=True)
