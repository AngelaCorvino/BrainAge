# pylint: disable=invalid-name, redefined-outer-name, import-error
"""
Main which finds optimum hyperparameters for a given model using grid-search. Compares different optimized regression models in pipeline for different harmonizing options.
Main which loads previously trained model and uses it to find the predicted age of a given dataframe.
"""
import pickle
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

from outliers import Outliers
from preprocessing import Preprocessing
from deepregression import DeepRegression
from crossvalidation import Crossvalidation

from matplotlib.offsetbox import AnchoredText
###############################################################OPTIONS
hyperparams = [
    {
        "Feature__k": [50, 100, 200, "all"],
        "Feature__score_func": [f_regression],
        "Model__epochs": [100, 200],
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
        "Model__kernel": ["linear", "rbf", "poly"],
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
    then fit the best performing model on the training set in cross validation .

    Parameters
    ----------

    dataframe : dataframe-like
                The dataframe of data to be passed to the function.

    model:      function-like
                The regression model to be passed to the function.


    hyparams:   dictionary-like
                A list of hyperparameters to be pssed to the function
                grid search  finsd the combination that generates the best result

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
        # if model.__class__.__name__== DeepRegression:
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
    # """
    #    We have our optimal list of parameters,
    #    we can run the model using these parameters.
    #    This cross validation is done using StratifiedKFold
    # """
    crossvalidation = Crossvalidation()
    model_fit, MSE, MAE, PR = crossvalidation.stratified_k_fold(
        x_train, y_train, y_train_class, 10, model_cv.best_estimator_
    )

    # Save the best performing model fitted in stratifiedkfold cross validation
    with open(
        "models/%s_%s_pkl" % (model.__class__.__name__, harmonize_option), "wb"
    ) as files:
        pickle.dump(model_fit, files)
<<<<<<< HEAD


=======
    # Save the metrics in txt_file
    header ='MSE\t'+'MAE\t'+'PR\t'
    metrics = [MSE, MAE, PR]
    metrics = np.array(metrics).T.tolist()
    np.savetxt('models/metrics/metrics_%s_%s_.txt' %(model.__class__.__name__, harmonize_option), metrics, header=header)
>>>>>>> 7b4e5b76928b46f844f59e4d1e7d72a4b22dab8b


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
    metric = np.genfromtxt('models/metrics/metrics_%s_%s_.txt' %(model.__class__.__name__, harmonize_option), metrics, header=header)

    with open(
        "models/%s_%s_pkl" % (model.__class__.__name__, harmonize_option), "rb"
    ) as f:
        model_fit = pickle.load(f)

    x_test = dataframe.drop(["AGE_AT_SCAN", "SEX", "DX_GROUP", "AGE_CLASS"], axis=1)
    y_test = dataframe["AGE_AT_SCAN"]

    predict_y = model_fit.predict(x_test)
    predict_y = np.squeeze(predict_y)
    delta = predict_y - y_test
    MSE = mean_squared_error(y_test, predict_y)
    MAE = mean_absolute_error(y_test, predict_y)
    PR = pearsonr(y_test, predict_y)[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        y_test, predict_y, alpha=0.5, c="y"
    )  # , c = MSE.map(colors), cmap = 'viridis'
    # plt.colorbar()
    plt.xlabel("Ground truth Age (years)", fontsize=18)
    plt.ylabel("Predicted Age (years)", fontsize=18)
    plt.plot(
        np.linspace(y_test.min(), predict_y.max(), 100),
        np.linspace(y_test.min(), predict_y.max(), 100),
        c="r",
        label="Expected prediction line",
    )
    text = AnchoredText(
    f" MAE= {round(MAE,3)} (years)\n MSE= {round(MSE,3)} (years)\n PR= {round(PR,3)}", prop=dict(size=14), frameon=True, loc='lower right')
    text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(text)
    plt.title(
        "Ground-truth Age versus Predict Age using \n \
            {}  with {} {} data".format(
            model.__class__.__name__, harmonize_option, prep.retrieve_name(dataframe)
        ),
        fontsize=20,
    )
    plt.tick_params(axis="x", which="major", labelsize=18)
    plt.tick_params(axis="y", which="major", labelsize=18)
    plt.legend()
    plt.savefig(
        "images/%s%s_%s.png"
        % (prep.retrieve_name(dataframe), model.__class__.__name__, harmonize_option),
        dpi=200,
        format="png",
    )
    return y_test, delta


def compare_prediction(y_test1, delta1, y_test2, delta2, model, harmonize_option):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test1, delta1, alpha=0.5, c="b")
    plt.scatter(y_test2, delta2, alpha=0.5, c="g")
    plt.xlabel("Ground truth Age (years)", fontsize=18)
    plt.ylabel("Delta Age (years)", fontsize=18)
    plt.title(
        "Delta Age versus Ground-truth  Age using \n \
            {}  with {} ".format(
            model.__class__.__name__, harmonize_option
        ),
        fontsize=20,
    )
    plt.tick_params(axis="x", which="major", labelsize=18)
    plt.tick_params(axis="y", which="major", labelsize=18)
    plt.legend()
    plt.savefig(
        "images/TDvsAS%s_%s.png" % (model.__class__.__name__, harmonize_option),
        dpi=200,
        format="png",
    )


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
            #"""
            #Tuning different regression model on healthy subjects
            #"""
            tune_model(df_TD_train, model, hyperparams[i], harmonize_option)

    for i, model in enumerate(models):
        age_truth_TD, delta_TD = predict_model(df_TD_test, model, harmonize_option)
        age_truth_AS, delta_AS = predict_model(df_AS, model, harmonize_option)
        compare_prediction(
            age_truth_TD, delta_TD, age_truth_AS, delta_AS, model, harmonize_option
        )
    print("You will find the saved images in Brainage/images")
