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


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import GridSearchCV

from crossvalidation import Crossvalidation
from features import Preprocessing
from deepregression import DeepRegression


# FUNCTIONS

def tune_model(dataframe, model, hyparams, harmonize_option):
    """
    Run run a grid-search for  hyperparameters tuning,
    then fit the model with the best performing model.

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

    (x_train, x_test, y_train, y_test, y_train_class, y_test_class,) = train_test_split(
        dataframe.drop(["AGE_AT_SCAN", "SEX", "DX_GROUP","AGE_CLASS"], axis=1),
        dataframe["AGE_AT_SCAN"],
        dataframe["AGE_CLASS"],
        test_size=0.25,
        random_state=18,
    )

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

    """
        We have our optimal list of parameters,
        we can run the model using these parameters.
        This cross validation is done using StratifiedKFold
    """

    model_fit, y_test, predict_y, MSE, MAE, PR = crossvalidation.stratified_k_fold(
        x_train, y_train, y_train_class, 10, model_cv.best_estimator_
    )
    """
    Save the best performing model fitted in stratifiedkfold cross validation

    """
    with open(
        "models/%s_%s_pkl" % (model.__class__.__name__, harmonize_option), "wb"
    ) as files:
        pickle.dump(model_fit, files)

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
        predict_y.max() - 12,
        f"MSE= {round(MSE,3)}",
        fontsize=14,
    )
    plt.text(
        y_test.max() - 18,
        predict_y.max() - 10,
        f"PR= {round(PR,3)}",
        fontsize=14,
    )
    plt.text(
        y_test.max() - 18, predict_y.max() - 14, f"MAE= {round(MAE,3)}", fontsize=14
    )
    plt.title(
        "Ground-truth Age versus Predict Age using \n \
            {}  with {} data".format(
            model.__class__.__name__, harmonize_option
        ),
        fontsize=20,
    )
    plt.tick_params(axis="x", which="major", labelsize=18)
    plt.tick_params(axis="y", which="major", labelsize=18)
    plt.legend(
        title="Number of features: {}".format(model_cv.best_params_["Feature__k"]),
        fontsize=18,
    )
    plt.savefig(
        "images/%s_%s.png" % (model.__class__.__name__, harmonize_option),
        dpi=200,
        format="png",
    )

########################################################PREPROCESSING
prep = Preprocessing()
df = prep.read_file("data/FS_features_ABIDE_males.csv")
crossvalidation = Crossvalidation()


models = [
    DeepRegression(),
    LinearRegression(),
    GaussianProcessRegressor(),
    RandomForestRegressor(),
    Lasso(),
    SVR(),
]


hyperparams = [
    {
        "Feature__k": [50,100, 200, "all"],
        "Feature__score_func": [f_regression],
        "Model__epochs": [100, 200],
        "Model__drop_rate": [0.2,0.4,0.6],
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
        "Model__random_state": [18],
    },
]
###############################################################################
harmonize_list = ["raw", "combat", "neuro"]
for harmonize_option in harmonize_list:
    """
    Compare different harmonization techniques
    """
    print("Harmonization model:", harmonize_option)
    dataframe = prep(df, harmonize_option, False)
    df_AS, df_TD = prep.split_file(dataframe)
    for i, model in enumerate(models):
        """
        Tuning different regression model on healthy subjects
        """
        tune_model(df_TD, model, hyperparams[i], harmonize_option)
