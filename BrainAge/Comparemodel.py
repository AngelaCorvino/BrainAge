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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

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
        print('Harmonization model:', harmonize_option)
        dataframe=prep(df, harmonize_option, False)
        df_AS, df_TD = file_split(dataframe)
        pipe = Pipeline(steps=[('Feature', SelectKBest(score_func=f_regression, k=10)),
                                ('Scaler', RobustScaler()),
                                ('regressionmodel', LinearRegression())])
        test_age,predict_age, MSE, MAE =regression.stratified_k_fold(df_TD.drop(['AGE_AT_SCAN'],axis=1), df_TD['AGE_AT_SCAN'], df_TD['AGE_CLASS'], 10, pipe)
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
        test_age, predict_age, MSE, MAE = regression.stratified_k_fold(df_TD.drop(['AGE_AT_SCAN'], axis=1), df_TD['AGE_AT_SCAN'], df_TD['AGE_CLASS'],                                                                         10,                                                                           pipe)
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



def run_randomforest(dataframe,harmonize_list):
    """
    Define dictonary in wihich searching the best set of hyperparameters
    """
    hyparams = {
    'Model__n_estimators': [200,300,400,500],
    'Model__max_features': ['sqrt', 'log2'],
    'Model__max_depth' : [4,5,6,7,8],
    'Model__random_state' : [18]
    }
    pipe = Pipeline(steps=[('Feature', SelectKBest(score_func=f_classif, k=10)),
                                    ('Scaler', RobustScaler()),
                                    ('Model',RandomForestRegressor())])
    for harmonize_option in harmonize_list:
        print('Harmonization model:',harmonize_option)
        dataframe=prep(df, harmonize_option,False)
        df_AS, df_TD = file_split(dataframe)

        x_train, x_test, y_train, y_test,y_train_class,y_test_class = train_test_split(df_TD.drop(['AGE_AT_SCAN'],axis=1), df_TD['AGE_AT_SCAN'],df_TD['AGE_CLASS'], test_size = .25, random_state = 18)
        rf_cv = GridSearchCV(pipe, cv=10, n_jobs=-1, param_grid=hyparams ,scoring='roc_auc')
        """
         If the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
        """
        #rf_cv.fit(df_TD.drop(['AGE_AT_SCAN']), df_TD['AGE_AT_SCAN'])
        rf_cv.fit(x_train, y_train_class)
        """
        Considering that the stratified kfold works better
        we coul decide to choose the  best hyperparameyer on age class.
        In this way we can use the stratified k-folf
        """
        print('Best parameters are:',rf_cv.best_params_.keys())

        """
        Now that we have our optimal list of parameters,
         we can run the basic RandomForestClassifier model using
         these parameters.
         This time we fir on age at scan, not age class
        """
        rf2 = RandomForestRegressor(rf_cv.best_params_).fit(x_train, y_train)
        predict_y=rf2.predict(x_test)
        MSE=mean_squared_error(y_test, predict_y, squared=False)
        plt.figure(figsize=(10,10))
        plt.scatter(y_test,predict_y,c='y')
        plt.xlabel('Ground truth Age(years)')
        plt.ylabel('Predicted Age(years)')
        plt.plot(np.linspace(test_age.min(),test_age.max(),100),np.linspace(test_age.min(),test_age.max(),100), c='r', label='Expected prediction line')
        plt.text(test_age.max()-2,predict_age.max()-2,f'Mean Absolute Error={MSE}',fontsize=14)
        plt.title('Ground-truth Age versus Predict Age using \n \
            Random Forest with {} harminization method'.format(harmonize_option))
        plt.show()
    return

#run_randomforest(df, ['raw'])
run_linearmodel(df, harmonize_list)

# #Deep learning
#
# deep = Deep(df_TD)
# deepmodel, deephistory = deep.make_autoencoder()
# deep.plot_training_validation_loss(deephistory)
# deep.reconstruction_error(deepmodel)
