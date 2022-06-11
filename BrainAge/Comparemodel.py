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
############################################################### FUNCTIONS
def file_split(dataframe):
        """
        Split dataframe in healthy (control) and autistic subjects groups
        """
        df_AS = dataframe.loc[dataframe.DX_GROUP == 1]
        df_TD = dataframe.loc[dataframe.DX_GROUP == -1]
        return df_AS, df_TD

########################################################PREPROCESSING
prep = Preprocessing()
df = prep.read_file("data/FS_features_ABIDE_males.csv")
regression = Regression()



models = [LinearRegression(), GaussianProcessRegressor(), RandomForestRegressor(), Lasso(), SVR()]

harmonize_list=['raw','combat','neuro']

def run_linearmodel(dataframe,harmonize_list ):
    for harmonize_option in harmonize_list:
        print('Harmonization model:',harmonize_option)
        dataframe=prep(df, harmonize_option,False)
        df_AS, df_TD = file_split(dataframe)
        pipe = Pipeline(steps=[('Feature', SelectKBest(score_func=f_classif, k=10)),
                                ('Scaler', RobustScaler()),
                                ('regressionmodel',LinearRegression())])
        test_age,predict_age, MSE, MAE =regression.stratified_k_fold(df_TD.drop(['AGE_AT_SCAN'],axis=1),
                                                        df_TD['AGE_AT_SCAN'],
                                                         df_TD['AGE_CLASS'],
                                                                            10,
                                                                            pipe)
        plt.figure(figsize=(10,10))
        plt.scatter(test_age,predict_age,c='y')
        plt.xlabel('Ground truth Age(years)')
        plt.ylabel('Predicted Age(years)')
        plt.plot(np.linspace(test_age.min(),test_age.max(),100),np.linspace(test_age.min(),test_age.max(),100), c='r', label='Expected prediction line')
        plt.text(test_age.max()-2,predict_age.max()-2,f'Mean Absolute Error={MSE}',fontsize=14)
        plt.title('Ground-truth Age versus Predict Age using \n \
            Linear Regression with {} harminization method'.format(harmonize_option))
        plt.show()
    return

def run_gaussianmodel(dataframe,harmonize_list ):
    for harmonize_option in harmonize_list:
        print('Harmonization model:',harmonize_option)
        dataframe=prep(df, harmonize_option,False)
        df_AS, df_TD = file_split(dataframe)
        pipe = Pipeline(steps=[('Feature', SelectKBest(score_func=f_classif, k=10)),
                                ('Scaler', RobustScaler()),
                                ('regressionmodel',GaussianProcessRegressor())])
        test_age,predict_age, MSE, MAE =regression.stratified_k_fold(df_TD.drop(['AGE_AT_SCAN'],axis=1),
                                                        df_TD['AGE_AT_SCAN'],
                                                         df_TD['AGE_CLASS'],
                                                                            10,
                                                                            pipe)
        plt.figure(figsize=(10,10))
        plt.scatter(test_age,predict_age,c='y')
        plt.xlabel('Ground truth Age(years)')
        plt.ylabel('Predicted Age(years)')
        plt.plot(np.linspace(test_age.min(),test_age.max(),100),np.linspace(test_age.min(),test_age.max(),100), c='r', label='Expected prediction line')
        plt.text(test_age.max()-2,predict_age.max()-2,f'Mean Absolute Error={MSE}',fontsize=14)
        plt.title('Ground-truth Age versus Predict Age using \n \
            Gaussian Regression with {} harminization method'.format(harmonize_option))
        plt.show()
    return


run_linearmodel(df, harmonize_list)


# #Deep learning
#
# deep = Deep(df_TD)
# deepmodel, deephistory = deep.make_autoencoder()
# deep.plot_training_validation_loss(deephistory)
# deep.reconstruction_error(deepmodel)
