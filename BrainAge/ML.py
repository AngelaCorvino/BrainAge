import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import verstack

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



from regression import Regression
from features import Utilities
#
a = Regression("data/FS_features_ABIDE_males.csv")

a.util.plot_histogram('AGE_AT_SCAN')
# a.util.plot_boxplot('Site', 'AGE_AT_SCAN', True)
# print(a.util.feature_selection('AGE_AT_SCAN', True).format())


models = [LinearRegression(), GaussianProcessRegressor(), RandomForestRegressor(), Lasso(), SVR()]

def run_models(models, model_results = []):
    a = Regression("data/FS_features_ABIDE_males.csv")
    a.rescale('Robust')
    for model in models:
        predict_age1, MSE1, MAE1 = a.k_Fold(10, model)
        predict_ag2, MSE2, MAE2 = a.Stratifiedk_Fold(10, model)
        model_results.append([MSE1, MAE1,MSE2,MAE2])

    return model_results
m=run_models(models)
