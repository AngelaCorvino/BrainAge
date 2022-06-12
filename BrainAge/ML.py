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
from sklearn.feature_selection import f_regression

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



def run_model(dataframe,model,hyparams):
    """
    Run run a grid-search for  hyperparaemters tuning,
    then run the model with the best combinatio of hyperparameters.

    Parameters
    ----------

    dataframe : dataframe-like
                The dataframe of data to be passed to the function.

    model:      function-like
                The regression model to be passed to the function.


    hyparams:   dictionary-like
                A list of hyperparameters to be pssed to the function
                grid search  finsd the combination that generates the best result


    """

    pipe = Pipeline(
        steps=[
            ("Feature", SelectKBest(score_func=f_regression)),
            ("Scaler", RobustScaler()),
            ("Model", model),
        ]
    )


    (
        x_train,
        x_test,
        y_train,
        y_test,
        y_train_class,
        y_test_class,
    ) = train_test_split(
        dataframe.drop(["AGE_AT_SCAN"], axis=1),
        dataframe["AGE_AT_SCAN"],
        dataframe["AGE_CLASS"],
        test_size=0.25,
        random_state=18,
        )
    model_cv = GridSearchCV(
            pipe, cv=10, n_jobs=-1, param_grid=hyparams, scoring="neg_mean_squared_error"
        )
    model_cv.fit(x_train, y_train)

    print("Best estimator is:", model_cv.best_estimator_)

    """
        Now that we have our optimal list of parameters,
        we can run the model using these parameters.
        This time the cross vazlidation is done using StratifiedKFold
    """
    y_test,predict_y, MSE, MAE = regression.stratified_k_fold(x_train,y_train, y_train_class, 5, model_cv.best_estimator_)

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
            f"MSE= {round(MSE,3)}",
            fontsize=14,
        )
    plt.text(y_test.max() - 20, predict_y.max() -22, f'MAE= {round(MAE,3)}',fontsize=14)
    plt.title(
            "Ground-truth Age versus Predict Age using \n \
            Gaussian Regression  with {} harmonization method".format(
                harmonize_option
            )
        )
    plt.show()

    return



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

for harmonize_option in harmonize_list:
    """
    Compare different harminization tecquiniques
    """
    print("Harmonization model:", harmonize_option)
    dataframe = prep(df, harmonize_option, False)
    df_AS, df_TD = file_split(dataframe)
    for model in models:
    """
    Compare different regression model
    """
        run_models(model,hyparams,df_TD)



#
# #Deep learning
#
# deep = Deep(df_TD)
# deepmodel, deephistory = deep.make_autoencoder()
# deep.plot_training_validation_loss(deephistory)
# deep.reconstruction_error(deepmodel)
# deep.outliers(deepmodel)
