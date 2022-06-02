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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from regression import Regression
from features import Preprocessing
#
#a = Regression("data/FS_features_ABIDE_males.csv")

#a.util.plot_histogram('AGE_AT_SCAN')
# a.util.plot_boxplot('Site', 'AGE_AT_SCAN', True)
# print(a.util.feature_selection('AGE_AT_SCAN', True).format())


models = [LinearRegression(), GaussianProcessRegressor(), RandomForestRegressor(), Lasso(), SVR()]
#
# def run_models(models, model_results = []):
#     a = Regression("data/FS_features_ABIDE_males.csv")
#     a.rescale('Robust')
#     for model in models:
#         predict_age1, MSE1, MAE1 = a.k_Fold(10, model)
#         predict_ag2, MSE2, MAE2 = a.Stratifiedk_Fold(10, model)
#         model_results.append([MSE1, MAE1,MSE2,MAE2])
#
#     return model_results
# m=run_models(models)


prep = Preprocessing()
df = prep.file_reader("data/FS_features_ABIDE_males.csv")
prep.add_features(df)
prep.add_binning(df)
y_bins = df['AGE_CLASS']
(df_AS, df_TD) = prep.file_split(df)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for model in models:
    pipe = Pipeline(steps=[('Feautureselection',prep.feature_selection(df_TD))
                        ('Scaler', RobustScaler()),
                      ('regressionmodel', model)])
    #pipe = pipe.fit(X.reshape(-1,1), y)
    pipe.fit(X_train, y_train)
    print(model)
    print("model score: %.3f" % pipe.score(X_test, y_test))
