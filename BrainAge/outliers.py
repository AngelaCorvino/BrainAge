# pylint: disable=invalid-name, redefined-outer-name
"""
Module  selects samples with reconstruction error larger than 3 sigmas and removes outlier samples from final dataframe.
"""
import pickle
import warnings
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit

from preprocessing import Preprocessing

warnings.filterwarnings("ignore")


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

    Attributes
    ----------

    dataframe : dataframe-like
        Dataframe to remove outliers on.
    model : object
        Trained RNN model.
    """

    def __init__(self, dataframe):
        """
        Constructor
        """
        self.dataframe = dataframe
        self.model = self.model_upload()

    def __call__(self, nbins, plot_fit=False, plot_distribution=False):
        """

        Parameters
        ----------

        nbins : integer-like
            Number of bins for loss histogram.
        plot_fit : boolean-like, default is False.
            If True shows histogram of loss of replicated data and gaussian fit for outlier detection.
        plot_distribution : boolean, default is False
            If True it shows the age distribution of removed samples.

        Returns
        -------
        dataframe : dataframe-like
            Dataframe without outliers.

        """
        indexes = self.outliers(nbins, plot_fit)
        if plot_distribution is True:
            self.plot_distribution(indexes, "AGE_AT_SCAN")
        clean = self.clean_dataframe(indexes)
        return clean

    def model_upload(self):
        """Uploads trained autoencoder model from file to run on dataframe.

        Returns
        -------
        model : object
            Trained RNN model

        """
        with open("models/autoencoder_pkl", "rb") as f:
            model = pickle.load(f)
        return model

    def outliers(self, nbins, plot_fit=False):
        """Identifies ouliers using autoencoder.

        Parameters
        ----------
        nbins : integer-like
            Number of bins for loss histogram.
        plot_fit : boolean-like, default is False.
            If True shows histogram of loss of replicated data and gaussian fit for outlier detection.

        Returns
        -------
        indexes : list-like
            List of indexes of samples to remove.
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
            x=test_mae_loss, bins=nbins, color="lightskyblue", label="Subjects"
        )
        plt.title("RNN Mean Absolute Error ", fontsize=24)
        plt.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.xlabel("MAE[years]", fontsize=24)
        plt.ylabel("N Subjects", fontsize=24)


        # Fit as a gaussian
        p0 = [0.2, 0.05, 1]

        fit, fitCov = curve_fit(gaussian, xdiscrete_1, n_1, p0=p0)
        fit_err = np.sqrt(abs(np.diag(fitCov)))
        print(
            f" Test fit parameters: \n x_0 = {fit[0] : .3f} +-{fit_err[0] : .3f}\n sigma = {fit[1] : .3f} +-{fit_err[1] : .3f}\n A = {fit[2] : .3f} +-{fit_err[2] : .3f}\n"
        )
        # Plot fit
        plt.plot(
            np.linspace(test_mae_loss.min(), test_mae_loss.max(), 1000),
            gaussian(np.linspace(test_mae_loss.min(), test_mae_loss.max(), 1000), *fit),
            color="green",
            linewidth=2,
            label="fit",
        )
        plt.axvspan(fit[0] - 3 * fit[1], fit[0] + 3 * fit[1],
        facecolor="green",
        alpha=0.1,label=r'$ x_0 \pm 3\sigma$')
        plt.legend(fontsize=20)
        if plot_fit is True:
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
        """Plots feature distribution of removed samples from dataframe.

        Parameters
        ----------
        indexes : list-like
            List of indexes of samples to remove as outliers.
        feature : string-like
            Feature to show in histogram

        """
        y = self.dataframe.iloc[indexes]
        bins = int(max(y[feature]) - min(y[feature]))
        plt.figure(2)
        plt.hist(
            x=y[feature],
            bins=bins,
            facecolor="lightskyblue",
            edgecolor="blue",
            linewidth=0.5,
            label="Subjects",
        )
        plt.title("Age Distribution of Outliers", fontsize=24)
        plt.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.xlabel("Age[years]", fontsize=22)
        plt.ylabel("N Subjects", fontsize=22)
        return plt.show()

    def clean_dataframe(self, indexes):
        """Removes sample of given indexes from dataframe.

        Parameters
        ----------
        indexes : list-like
            List of indexes of samples to remove.

        Returns
        -------
        dataframe
            Dataframe without outliers.
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
    out = Outliers(df_AS)
    df_AS= out(nbins=500, plot_fit=True, plot_distribution=True)
