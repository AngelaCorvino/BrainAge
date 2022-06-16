# pylint: disable=invalid-name, redefined-outer-name
"""
Module implements training in cross validation with K-fold and stratified K-fold.
"""
import numpy as np

from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


class Crossvalidation:
    """
    Class implementing model training in cross validation.
    """

    def k_fold(self, X, y, n_splits, model):
        """Fit and predict using cross validation
        with a model (or pipeline) supplied by the user.

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

        MSE = []
        MAE = []
        PR = []
        for train_index, validation_index in kf.split(X):
            predict_y = model.fit(X[train_index], y[train_index]).predict(X[validation_index])

            print("Model parameters:", model.get_params())

            MSE.append(mean_squared_error(y[validation_index], predict_y, squared=False))
            MAE.append(mean_absolute_error(y[validation_index], predict_y))
            PR.append(pearsonr(y[validation_index], predict_y))

        print(f"\n\nCross-Validation MSE, MAE: {np.mean(MSE):0.3f} {np.mean(MAE):0.3f}")

        return model, np.mean(MSE), np.mean(MAE), np.mean(PR)

    def stratified_k_fold(self, X, y, y_bins, n_splits, model):
        """Fit and predict using stratified cross validation
        with a model (or pipeline) supplied by the user.

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

        MSE = []
        MAE = []
        PR = []
        for train_index, validation_index in cv.split(X, y_bins):
            predict_y = model.fit(X[train_index], y[train_index]).predict(X[validation_index])
            print(f"MAE: {mean_absolute_error(y[validation_index], predict_y):0.3f}")
            MSE.append(mean_squared_error(y[validation_index], predict_y, squared=True))
            MAE.append(mean_absolute_error(y[validation_index], predict_y))
            y[validation_index] = np.squeeze(y[validation_index])
            predict_y = np.squeeze(predict_y)
            PR.append(pearsonr(y[validation_index], predict_y)[0])
        return (
            model,
            np.mean(MSE),
            np.mean(MAE),
            np.mean(PR, axis=0),
        )
