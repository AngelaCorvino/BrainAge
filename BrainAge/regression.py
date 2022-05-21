import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# from sklearn.preprocessing import PolynomialFeatures


from features import Utilities

class Regression():
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
        self.file_url = file_url
        self.util = Utilities(file_url)
        (self.df_AS, self.df_TD) = self.util.file_split()
        self.features = self.util.feature_selection('AGE_AT_SCAN', False)
        self.X,self.y=self.rescale()

    def rescale(self, scaler = None):
        """
        rescale data column-wise to have them in the same range
        """
        X = self.df_TD[self.features]
        y = self.df_TD['AGE_AT_SCAN']
        if scaler=='Standard':
            X = StandardScaler().fit_transform(X)
        if scaler== 'Robust':
            X = RobustScaler().fit_transform(X)
        else:
            pass
        return X,y

    def k_Fold(self,n_splits,model):
        """
        Split the data and test it on a model chosen by the user
        """

        try:
            self.y = self.y.to_numpy()
            self.X = self.X.to_numpy()
        except AttributeError:
            pass
        kf = KFold(n_splits)
        for train_index, test_index in kf.split(self.X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            # X_train, X_test = self.X[train_index],self. X[test_index]
            # y_train, y_test = self.y[train_index], self.y[test_index]
            predict_y= model.fit(self.X[train_index], self.y[train_index]).predict(self. X[test_index])
            MSE=mean_squared_error(self.y[test_index], predict_y, squared=False)
            MAE=mean_absolute_error(self.y[test_index], predict_y)
        return predict_y,MSE,MAE






if __name__ == "__main__":
    a=Regression("data/FS_features_ABIDE_males.csv")
    a.rescale('Robust')
    predict_age,MSE,MAE=a.k_Fold(10,LinearRegression())
    print(predict_age,MSE,MAE)

    predict_age,MSE,MAE=a.k_Fold(10,GaussianProcessRegressor())
    a.rescale('Standard')
    print(predict_age,MSE,MAE)
