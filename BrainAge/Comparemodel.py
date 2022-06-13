import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import verstack

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


models = [
    LinearRegression(),
    GaussianProcessRegressor(),
    RandomForestRegressor(),
    Lasso(),
    SVR(),
]

harmonize_list = ["raw", "combat", "neuro"]


def run_linearmodel(dataframe):
    """
    Define dictonary in which searching the best set of hyperparameters
    """
    hyparams = {"Feature__k": [10, 20, 30],
                "Feature__score_func":[f_regression]}
    pipe = Pipeline(
        steps=[
            ("Feature", SelectKBest()),
            ("Scaler", RobustScaler()),
            ("Model", LinearRegression()),
        ]
    )

    (x_train, x_test, y_train, y_test, y_train_class, y_test_class,) = train_test_split(
        dataframe.drop(["AGE_AT_SCAN"], axis=1),
        dataframe["AGE_AT_SCAN"],
        dataframe["AGE_CLASS"],
        test_size=0.25,
        random_state=18,
    )
    lr_cv = GridSearchCV(
        pipe, cv=10, n_jobs=-1, param_grid=hyparams, scoring="neg_mean_squared_error"
    )

    lr_cv.fit(x_train, y_train)

    print("Best estimator is:", lr_cv.best_estimator_)

    """
        Now that we have our optimal list of parameters,
        we can run the model using these parameters.
        This time the cross vazlidation is done using StratifiedKFold
    """
    y_test, predict_y, MSE, MAE = regression.stratified_k_fold(
        x_train, y_train, y_train_class, 5, lr_cv.best_estimator_
    )

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predict_y, c="y")
    plt.xlabel("Ground truth Age(years)")
    plt.ylabel("Predicted Age(years)")
    plt.plot(
        np.linspace(y_test.min(), predict_y.max(), 100),
        np.linspace(y_test.min(), predict_y.max(), 100),
        c="r",
        label="Expected prediction line",
    )
    plt.text(
            y_test.max() - 20,
            predict_y.max() - 20,
            f"MSE={round(MSE,3)}",
            fontsize=14,
        )
    plt.text(y_test.max() - 20, predict_y.max() -21, f'MAE ={round(MAE,3)}',fontsize=14)
    plt.title(
        "Ground-truth Age versus Predict Age using \n \
            Linear Regression with {} harmonization method".format(
            harmonize_option
        )
    )
    plt.show()

    return


def run_gaussianmodel(dataframe):
    """
    Define dictonary in which searching the best set of hyperparameters
    """
    hyparams = {
        # "Model__kernel": [200, 300, 400, 500],
        "Feature__k": [10, 20, 30],
        "Model__n_restarts_optimizer": [0, 1, 2],
        "Model__random_state": [18],
    }
    pipe = Pipeline(
        steps=[
            ("Feature", SelectKBest(score_func=f_regression)),
            ("Scaler", RobustScaler()),
            ("Model", GaussianProcessRegressor()),
        ]
    )

    (x_train, x_test, y_train, y_test, y_train_class, y_test_class,) = train_test_split(
        dataframe.drop(["AGE_AT_SCAN"], axis=1),
        dataframe["AGE_AT_SCAN"],
        dataframe["AGE_CLASS"],
        test_size=0.25,
        random_state=18,
    )
    gr_cv = GridSearchCV(
        pipe, cv=10, n_jobs=-1, param_grid=hyparams, scoring="neg_mean_squared_error"
    )

    gr_cv.fit(x_train, y_train)

    print("Best estimator is:", gr_cv.best_estimator_)

    """
        Now that we have our optimal list of parameters,
        we can run the model using these parameters.
        This time the cross validation is done using StratifiedKFold
    """
    y_test, predict_y, MSE, MAE = regression.stratified_k_fold(
        x_train, y_train, y_train_class, 5, gr_cv.best_estimator_
    )

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predict_y, c="y")
    plt.xlabel("Ground truth Age(years)")
    plt.ylabel("Predicted Age(years)")
    plt.plot(
        np.linspace(y_test.min(), predict_y.max(), 100),
        np.linspace(y_test.min(), predict_y.max(), 100),
        c="r",
        label="Expected prediction line",
    )
    plt.text(
            y_test.max() - 20,
            predict_y.max() - 20,
            f"MSE={round(MSE,3)}",
            fontsize=14,
        )
    plt.text(y_test.max() - 20, predict_y.max() -21, f'MAE ={round(MAE,3)}',fontsize=14)
    plt.title(
        "Ground-truth Age versus Predict Age using \n \
            Gaussian Regression  with {} harmonization method".format(
            harmonize_option
        )
    )
    plt.show()

    return


def run_randomforestmodel(dataframe):
    """
    Define dictonary in which searching the best set of hyperparameters
    """
    hyparams = {
        "Feature__k": [10, 20, 30],
        "Model__n_estimators": [10, 200, 300, 400],
        "Model__max_features": ["sqrt", "log2"],
        "Model__max_depth": [4, 5, 6, 7, 8],
        "Model__random_state": [18],
    }
    pipe = Pipeline(
        steps=[
            ("Feature", SelectKBest(score_func=f_regression)),
            ("Scaler", RobustScaler()),
            ("Model", RandomForestRegressor()),
        ]
    )
    (x_train, x_test, y_train, y_test, y_train_class, y_test_class,) = train_test_split(
        dataframe.drop(["AGE_AT_SCAN"], axis=1),
        dataframe["AGE_AT_SCAN"],
        dataframe["AGE_CLASS"],
        test_size=0.25,
        random_state=18,
    )
    rf_cv = GridSearchCV(
        pipe, cv=10, n_jobs=-1, param_grid=hyparams, scoring="neg_mean_squared_error"
    )
    rf_cv.fit(x_train, y_train)

    print("Best estimator is:", rf_cv.best_estimator_)

    """
        Now that we have our optimal list of parameters,
        we can run the basic RandomForestClassifier model using
        these parameters.
        At this point we can do cross validation with stratified_k_fold
    """
    y_test, predict_y, MSE, MAE = regression.stratified_k_fold(
            x_train, y_train, y_train_class, 5, rf_cv.best_estimator_
    )

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predict_y, c="y")
    plt.xlabel("Ground truth Age(years)")
    plt.ylabel("Predicted Age(years)")
    plt.plot(
            np.linspace(y_test.min(), predict_y.max(), 100),
            np.linspace(y_test.min(), predict_y.max(), 100),
            c="r",
            label="Expected prediction line",
        )
    plt.text(
            y_test.max() - 20,
            predict_y.max() - 20,
            f"MSE={round(MSE,3)}",
            fontsize=14,
        )
    plt.text(y_test.max() - 20, predict_y.max() -21, f'MAE ={round(MAE,3)}',fontsize=14)
    plt.title(
            "Ground-truth Age versus Predict Age using \n \
            Random Forest with {} data".format(
                harmonize_option
            )
        )
    plt.show()
    return


def run_lassomodel(dataframe):
    """
    Define dictonary in which searching the best set of hyperparameters
    """
    hyparams = {
        "Feature__k": [10, 20, 30],
        "Model__alpha": [ 0.1, 0.3, 0.6,1],
        "Model__random_state": [18],
    }
    pipe = Pipeline(
        steps=[
            ("Feature", SelectKBest(score_func=f_regression)),
            ("Scaler", RobustScaler()),
            ("Model", Lasso()),
        ]
    )
    (x_train, x_test, y_train, y_test, y_train_class, y_test_class,) = train_test_split(
        dataframe.drop(["AGE_AT_SCAN"], axis=1),
        dataframe["AGE_AT_SCAN"],
        dataframe["AGE_CLASS"],
        test_size=0.25,
        random_state=18,
    )
    la_cv = GridSearchCV(
        pipe, cv=10, n_jobs=-1, param_grid=hyparams, scoring="neg_mean_squared_error"
    )
    la_cv.fit(x_train, y_train)

    print("Best estimator is:", la_cv.best_estimator_)

    """
        Now that we have our optimal list of parameters,
        we can run the basic RandomForestClassifier model using
        these parameters.
        At this point we can do cross validation with stratified_k_fold
    """
    y_test, predict_y, MSE, MAE = regression.stratified_k_fold(
            x_train, y_train, y_train_class, 5, la_cv.best_estimator_
    )

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predict_y, c="y")
    plt.xlabel("Ground truth Age(years)")
    plt.ylabel("Predicted Age(years)")
    plt.plot(
            np.linspace(y_test.min(), predict_y.max(), 100),
            np.linspace(y_test.min(), predict_y.max(), 100),
            c="r",
            label="Expected prediction line",
        )
    plt.text(
            y_test.max() - 20,
            predict_y.max() - 20,
            f"MSE={round(MSE,3)}",
            fontsize=14,
        )
    plt.text(y_test.max() - 20, predict_y.max() -21, f'MAE ={round(MAE,3)}',fontsize=14)
    plt.title(
            "Ground-truth Age versus Predict Age using \n \
            Lasso model with {} data".format(
                harmonize_option
            )
        )
    plt.show()
    return




def run_SVRmodel(dataframe):
    """
    Define dictonary in which searching the best set of hyperparameters
    """
    hyparams = {
        "Feature__k": [10, 20, 30],
        "Model__kernel": ['linear', 'rbf','poly'],
        "Model__degree": [ 3, 4],
    }
    pipe = Pipeline(
        steps=[
            ("Feature", SelectKBest(score_func=f_regression)),
            ("Scaler", RobustScaler()),
            ("Model", SVR()),
        ]
    )
    (x_train, x_test, y_train, y_test, y_train_class, y_test_class,) = train_test_split(
        dataframe.drop(["AGE_AT_SCAN"], axis=1),
        dataframe["AGE_AT_SCAN"],
        dataframe["AGE_CLASS"],
        test_size=0.25,
        random_state=18,
    )
    svr_cv = GridSearchCV(
        pipe, cv=10, n_jobs=-1, param_grid=hyparams, scoring="neg_mean_squared_error"
    )
    svr_cv.fit(x_train, y_train)

    print("Best estimator is:", svr_cv.best_estimator_)

    """
        Now that we have our optimal list of parameters,
        we can run the basic RandomForestClassifier model using
        these parameters.
        At this point we can do cross validation with stratified_k_fold
    """
    y_test, predict_y, MSE, MAE = regression.stratified_k_fold(
            x_train, y_train, y_train_class, 5, svr_cv.best_estimator_
    )

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predict_y, c="y")
    plt.xlabel("Ground truth Age(years)")
    plt.ylabel("Predicted Age(years)")
    plt.plot(
            np.linspace(y_test.min(), predict_y.max(), 100),
            np.linspace(y_test.min(), predict_y.max(), 100),
            c="r",
            label="Expected prediction line",
        )
    plt.text(
            y_test.max() - 20,
            predict_y.max() - 20,
            f"MSE={round(MSE,3)}",
            fontsize=14,
        )
    plt.text(y_test.max() - 20, predict_y.max() -21, f'MAE ={round(MAE,3)}',fontsize=14)
    plt.title(
            "Ground-truth Age versus Predict Age using \n \
            SVR model with {} data".format(
                harmonize_option
            )
        )
    plt.show()
    return
###############################################################################
for harmonize_option in harmonize_list:
    print("Harmonization model:", harmonize_option)
    dataframe = prep(df, harmonize_option, False)
    df_AS, df_TD = file_split(dataframe)
    #run_linearmodel(df_TD)
    run_SVRmodel(df_TD)
    # run_gaussianmodel(df_TD)
    #run_randomforestmodel(df_TD)
