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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from regression import Regression
from features import Preprocessing
from deeplearning import Deep
def file_split(dataframe):
        """
        Split dataframe in healthy (control) and autistic subjects groups
        """
        df_AS = dataframe.loc[dataframe.DX_GROUP == 1]
        df_TD = dataframe.loc[dataframe.DX_GROUP == -1]
        return df_AS, df_TD


prep = Preprocessing()
df = prep.file_reader("data/FS_features_ABIDE_males.csv")
regression = Regression()

#SPLITTING DATA
(df_AS, df_TD) =file_split(prep(df, 'raw'))

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


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# for model in models:
#     pipe = Pipeline(steps=[('Feautureselection',prep.feature_selection(df_TD))
#                         ('Scaler', RobustScaler()),
#                       ('regressionmodel', model)])
#     #pipe = pipe.fit(X.reshape(-1,1), y)
#     pipe.fit(X_train, y_train)
#     print(model)
#     print("model score: %.3f" % pipe.score(X_test, y_test))



def run_models(models, model_results = []):
    for model in models:
        pipe = Pipeline(steps=[('Feauture',SelectKBest(score_func=f_classif, k=10)),('Scaler', RobustScaler()),
                          ('regressionmodel', model)])
        predict_age1, MSE1, MAE1 = regression.k_fold(df_TD.drop(['SITE','FILE_ID','AGE_AT_SCAN'],axis=1),df_TD['AGE_AT_SCAN'],10, pipe)
        predict_ag2, MSE2, MAE2 = regression.stratified_k_fold(df_TD.drop(['SITE','FILE_ID','AGE_AT_SCAN'],axis=1),df_TD['AGE_AT_SCAN'],df_TD['AGE_CLASS'],10, pipe)
        model_results.append([MSE1, MAE1,MSE2,MAE2])

    return model_results
m=run_models(models)
print(m)




#Deep learning

deep = Deep(df_TD)
deep.make_model().summary()
