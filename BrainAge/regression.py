import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from features import Utilities


class Regression():
    """
    Class describing regression model.
    Parameters
    ----------
    file_url : string-like
        The path that point to the data.

    """

    def __init__(self, file_url, features=0):
        """
        Constructor.
        """
        self.file_url = file_url
        self.util = Utilities(file_url)
        (self.df_AS, self.df_TD) = self.util.file_split()
        self.features = self.util.feature_selection('AGE_AT_SCAN', False)



if __name__ == "__main__":
    a=Regression("data/FS_features_ABIDE_males.csv")
