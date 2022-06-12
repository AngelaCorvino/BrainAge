import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from features import Preprocessing

class Regression:
    """
    Class describing regression model.
    """
    def k_fold(self, X, y, n_splits, model):
        """
        Splits the data and tests it on a model chosen by the user.

        :Parameters:

        X :
        y :
        n_splits :
        model :
        """
        try:
            y = y.to_numpy()
            X = X.to_numpy()
        except AttributeError:
            pass
        kf = KFold(n_splits)
        predict_age=[]
        MSE=[]
        MAE=[]
        for train_index, test_index in kf.split(X):

            # X_train, X_test = self.X[train_index],self. X[test_index]
            # y_train, y_test = self.y[train_index], self.y[test_index]

            predict_y=model.fit(X[train_index], y[train_index]).predict(X[test_index])
            print('Model parameters:',model.get_params())
            predict_age.append(predict_y)

            MSE.append(mean_squared_error(y[test_index], predict_y, squared=False))
            MAE.append(mean_absolute_error(y[test_index], predict_y))


        print('\n\nCross-Validation MSE, MAE: %.3f  %.3f' % (np.mean(MSE),np.mean(MAE)))

        return y[test_index],predict_y, np.mean(MSE), np.mean(MAE)

    def stratified_k_fold(self, X, y, y_bins, n_splits, model):
        """
        Split the data preserving distribution and test it on a model (or pipeline) chosen by the user.
        Parameters
        ----------
        X :
        y :
        y_bins :
        n_splits :
        model :
        """
        try:
            y = y.to_numpy()
            y_bins = y_bins.to_numpy()
            X = X.to_numpy()
        except AttributeError:
            pass
        cv = StratifiedKFold(n_splits)
        predict_age=[]
        MSE=[]
        MAE=[]
        for train_index, test_index in cv.split(X, y_bins):
            predict_y=model.fit(X[train_index], y[train_index]).predict(X[test_index])

            print('MSE:',mean_squared_error(y[test_index], predict_y, squared=False))
            MSE.append(mean_squared_error(y[test_index], predict_y, squared=False))
            MAE.append(mean_absolute_error(y[test_index], predict_y))

        return y[test_index],predict_y, np.mean(MSE), np.mean(MAE)

if __name__ == "__main__":
    prep = Preprocessing()
    df = prep.file_reader("data/FS_features_ABIDE_males.csv")

    dataframe=prep(df, 'raw')
    reg = Regression()
    model = LinearRegression()

    test_y,predict_y, MSE, MAE = reg.stratified_k_fold(X, y, prep(df, 'raw')['AGE_AT_SCAN'], 10, model)
