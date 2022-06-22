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
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

from matplotlib.offsetbox import AnchoredText

from outliers import Outliers
from preprocessing import Preprocessing
from deepregression import DeepRegression
from crossvalidation import Crossvalidation

warnings.filterwarnings("ignore")

###############################################################OPTIONS
hyperparams = [
    {
        "Feature__k": [ 128, 256, "all"],
        "Feature__score_func": [f_regression],
        "Model__epochs": [200, 300],
        "Model__drop_rate": [0.2, 0.4],
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
################################################# FUNCTIONS
def tune_model(dataframe_train, model, hyparams, harmonize_option):
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
    # """
    #    We have our optimal list of parameters,
    #    we can run the model using these parameters.
    #    This cross validation is done using StratifiedKFold
    # """
    crossvalidation = Crossvalidation()
    model_fit, MSE, MAE, PR = crossvalidation.stratified_k_fold(
        x_train, y_train, y_train_class, 10, model_cv.best_estimator_
    )
    # Save the metrics in txt_file
    header = "MSE\t" + "MAE\t" + "PR\t"
    metrics = np.array([MSE, MAE, PR])
    metrics = np.array(metrics).T
    np.savetxt(
        "models/metrics/metrics_%s_%s.txt"
        % (model.__class__.__name__, harmonize_option),
        metrics,
        header=header,
    )

    # Save the best performing model fitted in stratifiedkfold cross validation
    with open(
        "models/%s_%s_pkl" % (model.__class__.__name__, harmonize_option), "wb"
    ) as files:
        pickle.dump(model_fit, files)


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

    fig, ax = plt.subplots(figsize=(8, 8))
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
        fontsize=20,
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
        fontsize=20,
    )
    plt.tick_params(axis="x", which="major", labelsize=18)
    plt.tick_params(axis="y", which="major", labelsize=18)
    plt.legend(loc="upper right", fontsize=14)
    plt.savefig(
        "images/TDvsAS_%s_%s.png" % (model_name, harmonize_option),
        dpi=200,
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
prep = Preprocessing()
df = prep.read_file("data/FS_features_ABIDE_males.csv")
tune = input("Do you want to tune the models (yes/no): ")

for harmonize_option in harmonize_list:
    # """
    # Compare different harmonization techniques
    # """
    print("Harmonization model:", harmonize_option)
    dataframe = prep(df, harmonize_option, False)
    dataframe = prep.remove_strings(dataframe)
    df_AS, df_TD = prep.split_file(dataframe)
    out_td = Outliers(df_TD)
    df_TD = out_td(nbins=500, plot_fit=False, plot_distribution=False)
    out_as = Outliers(df_AS)
    df_AS = out_as(nbins=500, plot_fit=False, plot_distribution=False)
    (df_TD_train, df_TD_test) = train_test_split(
        df_TD,
        test_size=0.25,
        random_state=18,
    )

    for i, model in enumerate(models):
        if tune == "yes":
            # """
            # Tuning different regression model on healthy subjects
            # """
            tune_model(df_TD_train, model, hyperparams[i], harmonize_option)

        predict_age_TD_train, age_truth_TD_train, metric_train = predict_model(
            df_TD_train, model, harmonize_option
        )

        plot_model(
            predict_age_TD_train,
            age_truth_TD_train,
            model.__class__.__name__,
            get_name(df_TD_train),
            harmonize_option,
            metric_train,
        )

        predict_age_TD, age_truth_TD, metric_test_TD = predict_model(
            df_TD_test, model, harmonize_option
        )
        predict_age_AS, age_truth_AS, metric_AS = predict_model(
            df_AS, model, harmonize_option
        )

        plot_model(
            predict_age_TD,
            age_truth_TD,
            model.__class__.__name__,
            get_name(df_TD_test),
            harmonize_option,
            metric_test_TD,
        )

        plot_model(
            predict_age_AS,
            age_truth_AS,
            model.__class__.__name__,
            get_name(df_AS),
            harmonize_option,
            metric_AS,
        )
        compare_prediction(
            age_truth_TD,
            predict_age_TD,
            age_truth_AS,
            predict_age_AS,
            model.__class__.__name__,
            harmonize_option,
        )
    print("You will find the saved images in Brainage/images")
