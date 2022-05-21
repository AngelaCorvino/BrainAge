import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
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

    def k_Fold(self, n_splits, model):
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
        
        
""" Pretty functions"""       
   
def pretty(A, w=None, h=None):
    if A.ndim==1:
        if w == None :
            return str(A)
        else:
            s ='['+' '*(max(w[-1],len(str(A[0])))-len(str(A[0]))) +str(A[0])
            for i,AA in enumerate(A[1:]):
                s += ' '*(max(w[i],len(str(AA)))-len(str(AA))+1)+str(AA)
            s +='] '
    elif A.ndim==2:
        w1 = [max([len(str(s)) for s in A[:,i]])  for i in range(A.shape[1])]
        w0 = sum(w1)+len(w1)+1
        s= u'\u250c'+u'\u2500'*w0+u'\u2510' +'\n'
        for AA in A:
            s += ' ' + pretty(AA, w=w1) +'\n'    
        s += u'\u2514'+u'\u2500'*w0+u'\u2518'
    elif A.ndim==3:
        h=A.shape[1]
        s1=u'\u250c' +'\n' + (u'\u2502'+'\n')*h + u'\u2514'+'\n'
        s2=u'\u2510' +'\n' + (u'\u2502'+'\n')*h + u'\u2518'+'\n'
        strings=[pretty(a)+'\n' for a in A]
        strings.append(s2)
        strings.insert(0,s1)
        s='\n'.join(''.join(pair) for pair in zip(*map(str.splitlines, strings)))
    return s
    
def make_pretty(func): #esperimento con i decoratori
    def inner(A, *args):
        print((pretty((np.array(A)))))
        return func(A)
    return inner
    
"""Functions for future main"""

models = [LinearRegression(), GaussianProcessRegressor(), RandomForestRegressor(), Lasso(), SVR()]

def run_models(models, model_results = []):
    print(models)
    a = Regression("data/FS_features_ABIDE_males.csv")
    a.rescale('Robust')
    for model in models:
    	predict_age, MSE, MAE = a.k_Fold(10, model)
    	model_results.append([MSE, MAE])
    print(pretty((np.array(model_results))))
    return model_results


if __name__ == "__main__":
    run_models(models, [])
    #a=Regression("data/FS_features_ABIDE_males.csv")
    #a.rescale('Robust')
    #predict_age,MSE,MAE=a.k_Fold(10,LinearRegression())
    #print(predict_age,MSE,MAE)
    #predict_age,MSE,MAE=a.k_Fold(10,GaussianProcessRegressor())
    #a.rescale('Standard')
    #print(predict_age,MSE,MAE)
