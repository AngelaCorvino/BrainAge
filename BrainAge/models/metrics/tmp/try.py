import numpy as np

harmonize_list = ["normalized", "combat_harmonized", "neuro_harmonized"]

models = [
    # 'DeepRegression',
   'LinearRegression',
    'GaussianProcessRegressor',
    'RandomForestRegressor',
    'Lasso',
    'SVR',
]

for harmonize_option in harmonize_list:
    for i, model in enumerate(models):
        metric = np.genfromtxt(
            "../metrics_%s_%s.txt" % (model, harmonize_option),
            skip_header=1,
        )
        MSE, MAE, PR = np.mean(metric, axis=0)
        std_MSE, std_MAE, std_PR = np.std(metric, axis=0)
        header = "MSE\t" + "std_MSE\t" + "MAE\t" + "std_MAE\t" + "PR\t" + "std_PR\t"
        means = [f"{MSE} \\pm {std_MSE}  {MAE} \\pm {std_MAE} {PR} \\pm {std_PR}\n"]
        print("mean_%s_%s" %(model, harmonize_option))
        print(means)
