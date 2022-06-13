import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
# import verstack

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
from sklearn.model_selection import GridSearchCV

from regression import Regression
from features import Preprocessing
from deepregression import DeepRegression






# FUNCTIONS
def file_split(dataframe):
    """Split dataframe in healthy (control) and autistic subjects groups

    Parameters
    ----------
    dataframe : type
        Description of parameter `dataframe`.

    Returns
    -------
    type
        Description of returned object.

    """
    df_AS = dataframe.loc[dataframe.DX_GROUP == 1]
    df_TD = dataframe.loc[dataframe.DX_GROUP == -1]
    return df_AS, df_TD


def run_model(dataframe, model, harmonize_option):
    """


    Parameters
    ----------

    dataframe : dataframe-like
                The dataframe of data to be passed to the function.

    model:      function-like
                The regression model  to be passed to the function.



    harmonize_option: string-like


    """

    with open('models/%s_%s_pkl'%(model.__class__.__name__,harmonize_option) , 'rb') as f:
        model_fit= pickle.load(f)


    (x_train, x_test, y_train, y_test) = train_test_split(
        dataframe.drop(["AGE_AT_SCAN",'SEX','DX_GROUP'], axis=1),
        dataframe["AGE_AT_SCAN"],
        dataframe["AGE_CLASS"],
        test_size=1,
        random_state=18,
    )

    predict_y=model_fit.predict(x_test) # similar


    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predict_y, c="y")
    plt.xlabel("Ground truth Age(years)",fontsize=18)
    plt.ylabel("Predicted Age(years)",fontsize=18)
    plt.plot(
        np.linspace(y_test.min(), predict_y.max(), 100),
        np.linspace(y_test.min(), predict_y.max(), 100),
        c="r",
        label="Expected prediction line",
    )
    plt.text(
        y_test.max() - 18,
        predict_y.max() - 18,
        f"MSE= {round(MSE,3)}",
        fontsize=14,
    )
    plt.text(
        y_test.max() - 18,
        predict_y.max() - 16,
        f"PR= {round(PR,3)}",
        fontsize=14,
    )
    plt.text(
        y_test.max() - 18, predict_y.max() - 20, f"MAE= {round(MAE,3)}", fontsize=14
    )
    plt.title(
        "Ground-truth Age versus Predict Age using \n \
            {}  with {} data".format(model.__class__.__name__,
            harmonize_option),
            fontsize=20,
    )
    plt.tick_params(axis="x", which="major", labelsize=18)
    plt.tick_params(axis="y", which="major", labelsize=18)
    plt.legend(title='Number of features: {}'.format(model_cv.best_params_['Feature__k']),fontsize=18)
    plt.savefig('images/%s_%s.png'%(model.__class__.__name__,harmonize_option), dpi=200, format='png')
    #plt.show()

    return


########################################################PREPROCESSING
prep = Preprocessing()
df = prep.read_file("data/FS_features_ABIDE_males.csv")
regression = Regression()


models = [
    DeepRegression(),
    LinearRegression(),
    GaussianProcessRegressor(),
    RandomForestRegressor(),
    Lasso(),
    SVR(),
]

###############################################################################
harmonize_list = ["raw", "combat", "neuro"]

for harmonize_option in harmonize_list:
    """
    Compare different harmonization techniques
    """
    print("Harmonization model:", harmonize_option)
    dataframe = prep(df, harmonize_option, False)
    df_AS, df_TD = file_split(dataframe)
    for i, model in enumerate(models):
        """
        Predicting age of autistic subjects 
        """
        run_model(df_AS, model,harmonize_option)
