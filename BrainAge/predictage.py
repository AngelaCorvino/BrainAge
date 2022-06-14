import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


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

    with open(
        "models/%s_%s_pkl" % (model.__class__.__name__, harmonize_option), "rb"
    ) as f:
        model_fit = pickle.load(f)

    (_, x_test, _, y_test) = train_test_split(
        dataframe.drop(["AGE_AT_SCAN", "SEX", "DX_GROUP"], axis=1),
        dataframe["AGE_AT_SCAN"],
        test_size=0.9,
        random_state=18,
    )
    print(y_test)
    predict_y = model_fit.predict(x_test)  # similar
    MSE = mean_squared_error(y_test, predict_y)
    MAE = mean_absolute_error(y_test, predict_y)
    # PR=pearsonr(y_test,predict_y)[0]

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predict_y, c="y")
    plt.xlabel("Ground truth Age(years)", fontsize=18)
    plt.ylabel("Predicted Age(years)", fontsize=18)
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
    # plt.text(
    #     y_test.max() - 18,
    #     predict_y.max() - 16,
    #     f"PR= {round(PR,3)}",
    #     fontsize=14,
    # )
    plt.text(
        y_test.max() - 18, predict_y.max() - 20, f"MAE= {round(MAE,3)}", fontsize=14
    )
    plt.title(
        "Ground-truth Age versus Predict Age using \n \
            {}  with {} AD   q12data".format(
            model.__class__.__name__, harmonize_option
        ),
        fontsize=20,
    )
    plt.tick_params(axis="x", which="major", labelsize=18)
    plt.tick_params(axis="y", which="major", labelsize=18)
    plt.legend()
    plt.savefig(
        "images/AD%s_%s.png" % (model.__class__.__name__, harmonize_option),
        dpi=200,
        format="png",
    )



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
        run_model(df_AS, model, harmonize_option)
