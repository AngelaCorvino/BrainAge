# pylint: disable=invalid-name
import numpy as np

from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from features import Preprocessing


class Regression:
    """
    Class describing regression model.
    """

    def k_fold(self, X, y, n_splits, model):
        """Splits the data and tests it on a model chosen by the user.

        Parameters
        ----------
        X : type
            Description of parameter `X`.
        y : type
            Description of parameter `y`.
        n_splits : type
            Description of parameter `n_splits`.
        model : type
            Description of parameter `model`.

        Returns
        -------
        type
            Description of returned object.

        """
        try:
            y = y.to_numpy()
            X = X.to_numpy()
        except AttributeError:
            pass
        kf = KFold(n_splits)
        predict_age = []
        MSE = []
        MAE = []
        PR = []
        for train_index, test_index in kf.split(X):

            # X_train, X_test = self.X[train_index],self. X[test_index]
            # y_train, y_test = self.y[train_index], self.y[test_index]

            predict_y = model.fit(X[train_index], y[train_index]).predict(X[test_index])
            print("Model parameters:", model.get_params())
            predict_age.append(predict_y)

            MSE.append(mean_squared_error(y[test_index], predict_y, squared=False))
            MAE.append(mean_absolute_error(y[test_index], predict_y))
            PR.append(pearsonr(y[test_index], predict_y))

        print(
            "\n\nCross-Validation MSE, MAE: %.3f  %.3f" % (np.mean(MSE), np.mean(MAE))
        )

        return y[test_index], predict_y, np.mean(MSE), np.mean(MAE), np.mean(PR)

    def stratified_k_fold(self, X, y, y_bins, n_splits, model):
        """Split the data preserving distribution and test it
        on a model (or pipeline) chosen by the user.

        Parameters
        ----------
        X : type
            Description of parameter `X`.
        y : type
            Description of parameter `y`.
        y_bins : type
            Description of parameter `y_bins`.
        n_splits : type
            Description of parameter `n_splits`.
        model : type
            Description of parameter `model`.

        Returns
        -------
        type
            Description of returned object.

        """

        try:
            y = y.to_numpy()
            y_bins = y_bins.to_numpy()
            X = X.to_numpy()
        except AttributeError:
            pass
        cv = StratifiedKFold(n_splits)
        predict_age = []
        MSE = []
        MAE = []
        PR = []
        for train_index, test_index in cv.split(X, y_bins):
            predict_y = model.fit(X[train_index], y[train_index]).predict(X[test_index])
            print("MAE: %.3f" % mean_absolute_error(y[test_index], predict_y))
            MSE.append(mean_squared_error(y[test_index], predict_y, squared=True))
            MAE.append(mean_absolute_error(y[test_index], predict_y))
            PR.append(pearsonr(y[test_index],predict_y)[0])
        return (
            model,
            y[test_index],
            predict_y,
            np.mean(MSE),
            np.mean(MAE),
            np.mean(PR,axis=0))


if __name__ == "__main__":
    def file_split(dataframe):
        """
        Split dataframe in healthy (control) and autistic subjects groups
        """
        df_AS = dataframe.loc[dataframe.DX_GROUP == 1]
        df_TD = dataframe.loc[dataframe.DX_GROUP == -1]
        return df_AS, df_TD

    prep = Preprocessing()
    df = prep.read_file("data/FS_features_ABIDE_males.csv")
    df = prep(df, "combat")
    df_TD, df_ASD = file_split(df)
    reg = Regression()
    model = LinearRegression()
    model_fit,test_y, predict_y, MSE, MAE, PR = reg.stratified_k_fold(
        df_TD.drop(["AGE_AT_SCAN"], axis=1),
        df_TD["AGE_AT_SCAN"],
        df_TD["AGE_CLASS"],
        10,
        model,
    )
    print(PR)
