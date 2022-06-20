# pylint: disable=invalid-name, redefined-outer-name, import-error
"""
Main which finds optimum hyperparameters for a given model using grid-search. Compares different optimized regression models in pipeline for different harmonizing options.
Main which loads previously trained model and uses it to find the predicted age of a given dataframe.
"""
import pickle
import warnings
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

from matplotlib.offsetbox import AnchoredText

from preprocessing import Preprocessing
from deepregression import DeepRegression

warnings.filterwarnings("ignore")


###############################################################OPTIONS
hyperparams = [
    {
        "Feature__k": [50, 100, 200, "all"],
        "Feature__score_func": [f_regression],
        "Model__epochs": [200, 300],
        "Model__drop_rate": [0.2, 0.4, 0.6],
    },
    {
        "Feature__k": [10, 20, 30],
        "Feature__score_func": [f_regression],
    },
    {
        "Feature__k": [10, 20, 30],
        "Feature__score_func": [f_regression],
        "Model__n_restarts_optimizer": [0, 1, 2],
        "Model__random_state": [18],
    },
    {
        "Feature__k": [10, 20, 30],
        "Feature__score_func": [f_regression],
        "Model__n_estimators": [10, 200, 300],
        "Model__max_features": ["sqrt", "log2"],
        "Model__max_depth": [4, 5, 6, 7, 8],
        "Model__random_state": [18],
    },
    {
        "Feature__k": [10, 20, 30],
        "Feature__score_func": [f_regression],
        "Model__alpha": [0.1, 0.3, 0.6, 1],
        "Model__random_state": [18],
    },
    {
        "Feature__k": [10, 20, 30],
        "Feature__score_func": [f_regression],
        "Model__kernel": ["rbf", "poly"],
        "Model__degree": [3, 4],
    },
]

harmonize_list = ["normalized", "combat_harmonized", "neuro_harmonized"]

models = [
    DeepRegression(plot_loss=False),
    LinearRegression(),
    GaussianProcessRegressor(),
    RandomForestRegressor(),
    Lasso(),
    SVR(),
]
#######################################################FUNCTIONS
def tune_model(
    dataframe_train, model, hyparams, harmonize_option
):  # Questo tune è diverso da quello in predictage, è sperimentale e senza cross-validation
    """
    Run a grid-search for  hyperparameters tuning,
    then fit the best performing model on the training set in cross validation.

    Parameters
    ----------

    dataframe : dataframe-like
                The dataframe of data to be passed to the function.

    model:      function-like
                The regression model to be passed to the function.


    hyparams:   dictionary-like
                A list of hyperparameters to be passed to the Grid search.

    harmonize_option: string-like


    """

    pipe = Pipeline(
        steps=[
            ("Feature", SelectKBest()),
            ("Scaler", RobustScaler()),
            ("Model", model),
        ]
    )

    x_train = dataframe_train.drop(
        ["AGE_AT_SCAN", "SEX", "DX_GROUP", "AGE_CLASS"], axis=1
    )
    y_train = dataframe_train["AGE_AT_SCAN"]
    y_train_class = dataframe_train["AGE_CLASS"]

    if model == DeepRegression():
        print("No cross validation for deep model")
        model_cv = GridSearchCV(
            pipe,
            cv=None,
            n_jobs=-1,
            param_grid=hyparams,
            scoring="neg_mean_absolute_error",
            verbose=True,
        )
    else:
        print("Cross validation for regression model")
        model_cv = GridSearchCV(
            pipe,
            cv=10,
            n_jobs=-1,
            param_grid=hyparams,
            scoring="neg_mean_absolute_error",
            verbose=True,
        )

    model_cv.fit(x_train, y_train)

    print("Best combination of hyperparameters:", model_cv.best_params_)

    # Save the best performing model fitted in stratifiedkfold cross validation
    with open(
        "models/%s_%s_pkl" % (model.__class__.__name__, harmonize_option), "wb"
    ) as files:
        pickle.dump(model_cv, files)


def predict_model(dataframe, model, harmonize_option):
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
    x_test = dataframe.drop(["AGE_AT_SCAN", "SEX", "DX_GROUP", "AGE_CLASS"], axis=1)
    y_test = dataframe["AGE_AT_SCAN"]

    predict_y = model_fit.predict(x_test)
    predict_y = np.squeeze(predict_y)
    metric_test = np.array(
        [
            mean_squared_error(y_test, predict_y),
            mean_absolute_error(y_test, predict_y),
            pearsonr(y_test, predict_y)[0],
        ]
    )
    return predict_y, y_test, metric_test


def plot_model(predict_y, y_test, model_name, dataframe_name, harmonize_option, metric):

    if metric.ndim == 1:
        MSE, MAE, PR = metric[0], metric[1], metric[2]
        text = AnchoredText(
            f"Test Dataset \n MAE= {round(MAE,3)} [years]\n MSE= {round(MSE,3)} [years]\n PR= {round(PR,3)}",
            prop=dict(size=14),
            frameon=True,
            loc="lower right",
        )
    elif metric.ndim == 2:
        MSE, MAE, PR = np.mean(metric, axis=0)
        std_MSE, std_MAE, std_PR = np.std(metric, axis=0)
        text = AnchoredText(
            f" Train Dataset \n MAE = {round(np.mean(MAE),3)} +- {round(std_MAE,3)} [years] \n MSE = {round(MSE,3)} +- {round(std_MSE,3)} [years] \n PR = {round(PR,3)} +- {round(std_PR,3)}",
            prop=dict(size=14),
            frameon=True,
            loc="lower right",
        )

    _, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        y_test,
        predict_y,
        alpha=0.5,
        c="y",
        label=f"{dataframe_name}",
    )
    plt.xlabel("Ground truth Age [years]", fontsize=18)
    plt.ylabel("Predicted Age [years]", fontsize=18)
    plt.plot(
        np.linspace(y_test.min(), predict_y.max(), 100),
        np.linspace(y_test.min(), predict_y.max(), 100),
        c="r",
        label="Expected prediction",
    )

    text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(text)
    plt.title(
        "Ground-truth Age versus Predict Age using \n \
            {}  with {} data".format(
            model_name, harmonize_option
        ),
        fontsize=24,
    )
    plt.tick_params(axis="x", which="major", labelsize=18)
    plt.tick_params(axis="y", which="major", labelsize=18)
    plt.legend(loc="upper right", fontsize=14)

    plt.savefig(
        "images/%s_%s_%s.png"
        % (dataframe_name, model.__class__.__name__, harmonize_option),
        dpi=200,
        format="png",
        bbox_inches="tight",
    )


def compare_prediction(
    y_test1, predict_y1, y_test2, predict_y2, model_name, harmonize_option
):
    """Compare prediction performances of the same model on two different dataset.

    Parameters
    ----------
    y_test1 : array-like
        Test feature from the first data set.
    predict_y1 : array-like
        Predicted feauture from the first data set.
    y_test2 : array-like
        Test feature from the second data set.
    predict_y2 : type
        Predicted feature from the second data set.
    model_name : string-like
        Name of the model used for prediction.
    harmonize_option : string-like
        Harmonization method applied on data set.

    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test1, predict_y1 - y_test1, alpha=0.5, c="b", label="dataset1")
    plt.scatter(y_test2, predict_y2 - y_test2, alpha=0.5, c="g", label="dataset2")
    plt.xlabel("Ground truth Age [years]", fontsize=18)
    plt.ylabel("Delta Age [years]", fontsize=18)
    plt.title(
        "Delta Age versus Ground-truth  Age using \n \
            {}  with {} ".format(
            model_name,
            harmonize_option,
        ),
        fontsize=24,
    )
    plt.tick_params(axis="x", which="major", labelsize=18)
    plt.tick_params(axis="y", which="major", labelsize=18)
    plt.legend(loc="upper right", fontsize=14)
    plt.savefig(
        "images/TDvsAS_%s_%s.png" % (model_name, harmonize_option),
        dpi=240,
        format="png",
    )


def get_name(dataframe):
    """Retrieves name of variable.

    Parameters
    ----------
    var : type
        Variable of which we want the name.

    Returns
    -------
    prep.retrieve_name(dataframe) : string-like
        Name of the variable.

    """
    return prep.retrieve_name(dataframe)


##################################################MAIN
sites = [
    "Caltech",
    "CMU",
    "KKI",
    "Leuven",
    "MaxMun",
    "NYU",
    "OHSU",
    "Olin",
    "Pitt",
    "SBL",
    "SDSU",
    "Stanford",
    "Trinity",
    "UCLA",
    "UM",
    "USM",
    "Yale",
]
len_sites = len(sites)
site_dict = {
    1: "Caltech",
    2: "CMU",
    3: "KKI",
    4: "Leuven",
    5: "MaxMun",
    6: "NYU",
    7: "OHSU",
    8: "Olin",
    9: "Pitt",
    10: "SBL",
    11: "SDSU",
    12: "Stanford",
    13: "Trinity",
    14: "UCLA",
    15: "UM",
    16: "USM",
    17: "Yale",
}

# col_dict = {1: 'blue', 2: 'red', 3: 'green', 4: 'yellow', 5: 'cyan', 6: 'palegreen', 7: 'deepskyblue', 8: 'orange', 9: 'pink', 10: 'orchid', 11: 'royalblue', 12: 'salmon', 13: 'purple', 14: 'olive', 15: 'lightseagreen', 16: 'dodgerblue', 17: 'teal'}

prep = Preprocessing()

cm = plt.get_cmap("viridis")
color = [cm(1.0 * i / len_sites) for i in range(len_sites)]

for harmonize_option in harmonize_list:
    df = prep.read_file("data/FS_features_ABIDE_males.csv")
    s = []
    df, s = prep(df, harmonize_option, False, site_dfs=True)
    #df = prep.remove_strings(df)
    for model in models:
        #_, df_TD=prep.split_file(df)
        #tune_model(df_TD, model, hyperparams, harmonize_option)
        s_TD = [0]
        s_AS = [0]
        MAE = [0]
        fig, ax = plt.subplots(figsize=(17, 10))
        for i in range(1, len_sites + 1):
            try:
                assert (
                    np.sum(np.sum(s[i - 1].isna())) == 0
                ), "There are NaN values in the dataframe!"
            except AssertionError as msg:
                print(msg)
            s_AS.append(list(prep.split_file(s[i - 1]))[0])
            s_TD.append(list(prep.split_file(s[i - 1]))[1])
            predict_age, age_truth, metric = predict_model(
                s_TD[i].drop(["SITE", "SITE_CLASS", "FILE_ID"], axis=1),
                model,
                harmonize_option,
            )
            ax.scatter(
                age_truth,
                predict_age,
                alpha=0.7,
                color=color[i - 1],
                label=site_dict[i],
            )
            MAE.append(metric[1])
        MAE.pop(0)
        std_MAE = np.std(MAE)
        print(f"Standard deviation is {std_MAE}" + " for " + harmonize_option)
        plt.title(
            f"Ground-truth Age versus Predict Age using {harmonize_option} data",
            fontsize=28,
        )
        plt.xlabel("Ground truth Age [years]", fontsize=28)
        plt.ylabel("Predicted Age [years]", fontsize=28)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.plot(
            np.linspace(5, 60, 100),
            np.linspace(5, 60, 100),
            alpha=(0.5),
            c="r",
            label="Expected prediction",
        )
        plt.legend(fontsize=18)
        plt.tight_layout()
        plt.savefig(
            "images/by_site/TD_%s_%s_by_site.png"
            % (model.__class__.__name__, harmonize_option),
            dpi=240,
            format="png",
        )
        fig1, ax1 = plt.subplots(figsize=(24, 12))
        x = np.arange(len_sites)
        plt.bar(x, height=MAE, color="lightskyblue", label="Subjects")
        plt.xticks(x, sites[0:len_sites], fontsize=24)
        plt.yticks(fontsize=24)
        plt.title(
            "Mean Absolute Error by Site for %s" % (harmonize_option), fontsize=28
        )
        plt.ylabel("MAE [years]", fontsize=28)
        text = AnchoredText(
            "MAE = %.3f [years] \nMAE Standard Deviation= %.3f [years]"
            % (np.mean(MAE), std_MAE),
            prop=dict(size=28),
            frameon=True,
            loc="upper right",
        )
        text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax1.add_artist(text)
        plt.tight_layout()
        plt.savefig(
            "images/by_site/MAE_%s_%s_by_site.png"
            % (model.__class__.__name__, harmonize_option),
            dpi=240,
            format="png",
        )
