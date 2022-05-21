import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from regression import Regression
from features import Utilities

a = Regression("data/FS_features_ABIDE_males.csv")
#a.util.add_features()
#a.util.df_AS, a.util.df_TD = a.util.file_split()
print(a.util.df_TD.shape)
print(a.util.df_AS.shape)
a.util.plot_histogram('AGE_AT_SCAN')
a.util.plot_boxplot('Site', 'AGE_AT_SCAN', True)
print(a.util.feature_selection('AGE_AT_SCAN', True).format())
