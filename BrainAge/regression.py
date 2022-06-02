import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# from neurocombat_sklearn import CombatModel

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from neuroCombat import neuroCombat

from features import Preprocessing


class Regression:
    """
    Class describing regression model.
    Parameters
    ----------
    file_url : string-like
        The path that point to the data.

    """

    def __init__(self, file_url, features=0):
        """
        Constructor.
        """

    def rescale(self, scaler=None):
        """
        Rescale data column-wise to have them in the same range
        """
        if scaler == "Standard":
            X = StandardScaler().fit_transform(X)
        if scaler == "Robust":
            X = RobustScaler().fit_transform(X)
        else:
            pass
        return X, y

    def k_fold(self, X, y, n_splits, model):
        """
        Split the data and test it on a model chosen by the user
        """
        try:
            y = y.to_numpy()
            X = X.to_numpy()
        except AttributeError:
            pass
        kf = KFold(n_splits)
        for train_index, test_index in kf.split(X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            # X_train, X_test = self.X[train_index],self. X[test_index]
            # y_train, y_test = self.y[train_index], self.y[test_index]
            predict_y = model.fit(X[train_index], y[train_index]).predict(X[test_index])
            MSE = mean_squared_error(y[test_index], predict_y, squared=False)
            MAE = mean_absolute_error(y[test_index], predict_y)
        return predict_y, MSE, MAE

    def stratified_k_fold(self, X, y, y_bins, n_splits, model):
        """
        Split the data preserving distribution and test it on a model chosen by the user
        """
        try:
            y = y.to_numpy()
            y_bins = y_bins.to_numpy()
            X = X.to_numpy()
        except AttributeError:
            pass
        cv = StratifiedKFold(n_splits)
        for train_index, test_index in cv.split(X, y_bins):
            predict_y = model.fit(X[train_index], y[train_index]).predict(X[test_index])
            MSE = mean_squared_error(y[test_index], predict_y, squared=False)
            MAE = mean_absolute_error(y[test_index], predict_y)

        return predict_y, MSE, MAE


if __name__ == "__main__":
    prep = Preprocessing()
    df = prep.file_reader("data/FS_features_ABIDE_males.csv")
    prep.add_features(df)
    prep.add_binning(df)
    y_bins = df['AGE_CLASS']
    prep.plot_histogram(df, 'AGE_AT_SCAN')
    (df_AS, df_TD) = prep.file_split(df)
    prep.plot_boxplot(df_TD, 'Site', 'AGE_AT_SCAN')
    features, X, y = prep.feature_selection(df_TD)
    harmonization = False
    if harmonization == True:
        df_TD = prep.com_harmonization(df_TD)
        print(df_TD)
    reg = Regression("data/FS_features_ABIDE_males.csv")
    model = LinearRegression()
    stratified = True
    if stratified == True:
        predict_y, MSE, MAE = reg.stratified_k_fold(X, y, y_bins, 10, model)
    else: 
        predict_y, MSE, MAE = reg.k_Fold(X, y, 10, model)
